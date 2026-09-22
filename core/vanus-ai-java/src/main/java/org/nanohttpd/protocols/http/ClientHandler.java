/*
 * #%L
 * NanoHttpd-Core
 * %%
 * Copyright (C) 2012 - 2016 nanohttpd
 * %%
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the nanohttpd nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
package org.nanohttpd.protocols.http;


import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.util.logging.Level;

import org.nanohttpd.protocols.http.tempfiles.ITempFileManager;

/**
 * The runnable that will be used for every new client connection.
 */
public class ClientHandler implements Runnable {

    private final NanoHTTPD httpd;

    private final InputStream inputStream;

    private final Socket acceptSocket;

    public ClientHandler(NanoHTTPD httpd, InputStream inputStream, Socket acceptSocket) {
        this.httpd = httpd;
        this.inputStream = inputStream;
        this.acceptSocket = acceptSocket;
    }

    public enum Phase { IDLE, HEADERS, HANDLER, BODY, WRITE }

    private volatile long deadlineNanos = Long.MAX_VALUE;
    private volatile long requestDeadlineNanos = Long.MAX_VALUE;
    private volatile Phase phase = Phase.IDLE;
    private volatile boolean draining;
    private volatile Thread worker;

    public void close() {
        // Close the socket first: closing a buffered stream can wait on a blocked reader.
        NanoHTTPD.safeClose(this.acceptSocket);
        Thread thread = worker;
        if (thread != null && thread != Thread.currentThread()) thread.interrupt();
    }

    public synchronized void beginPhase(Phase next) {
        phase = next;
        HttpLimits limits = httpd.getLimits();
        long now = System.nanoTime();
        int timeout = switch (next) {
            case IDLE -> limits.idleTimeoutMillis();
            case HEADERS -> limits.headerTimeoutMillis();
            case HANDLER -> limits.requestTimeoutMillis();
            case BODY -> limits.bodyTimeoutMillis();
            case WRITE -> limits.writeTimeoutMillis();
        };
        if (next == Phase.HANDLER) requestDeadlineNanos = now + timeout * 1_000_000L;
        if (next == Phase.IDLE) requestDeadlineNanos = Long.MAX_VALUE;
        deadlineNanos = now + timeout * 1_000_000L;
        if (next == Phase.BODY || next == Phase.HANDLER) {
            deadlineNanos = Math.min(deadlineNanos, requestDeadlineNanos);
        }
        if (draining && (next == Phase.IDLE || next == Phase.HEADERS)) close();
    }

    public synchronized void expireIfOverdue(long nowNanos) {
        if (deadlineNanos != Long.MAX_VALUE && nowNanos - deadlineNanos >= 0) close();
    }

    public synchronized void drain() {
        draining = true;
        if (phase == Phase.IDLE || phase == Phase.HEADERS) close();
    }

    @Override
    public void run() {
        worker = Thread.currentThread();
        OutputStream outputStream = null;
        try {
            outputStream = this.acceptSocket.getOutputStream();
            ITempFileManager tempFileManager = httpd.getTempFileManagerFactory().create();
            HTTPSession session = new HTTPSession(httpd, tempFileManager, this.inputStream,
                    outputStream, this.acceptSocket.getInetAddress());
            session.setPhaseListener(this::beginPhase);
            int requests = 0;
            while (!this.acceptSocket.isClosed() && !draining) {
                beginPhase(Phase.IDLE);
                session.setLastRequest(++requests >= httpd.getLimits().maxRequestsPerConnection());
                session.execute();
            }
        } catch (Exception e) {
            if (!(e instanceof SocketException) && !(e instanceof SocketTimeoutException)
                    && !(e instanceof InterruptedException)) {
                NanoHTTPD.LOG.log(Level.WARNING, "Connection handler failed", e);
            }
        } finally {
            NanoHTTPD.safeClose(this.acceptSocket);
            NanoHTTPD.safeClose(outputStream);
            NanoHTTPD.safeClose(this.inputStream);
            worker = null;
            httpd.asyncRunner.closed(this);
        }
    }
}
