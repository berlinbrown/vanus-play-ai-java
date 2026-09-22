package org.nanohttpd.protocols.http;

/** Resource budgets shared by the parser and connection lifecycle. All times are milliseconds. */
public record HttpLimits(
        int maxHeaderBytes, long maxBodyBytes, int maxMultipartParts,
        int maxRequestsPerConnection, int idleTimeoutMillis, int headerTimeoutMillis,
        int bodyTimeoutMillis, int requestTimeoutMillis, int writeTimeoutMillis) {
    public static final HttpLimits DEFAULT = new HttpLimits(
            8192, 1024 * 1024, 32, 100, 5000, 10000, 15000, 30000, 30000);

    public HttpLimits {
        if (maxHeaderBytes < 256 || maxHeaderBytes > 65536 || maxBodyBytes < 0
                || maxBodyBytes > 16 * 1024 * 1024 || maxMultipartParts < 1
                || maxRequestsPerConnection < 1 || idleTimeoutMillis < 1
                || headerTimeoutMillis < 1 || bodyTimeoutMillis < 1
                || requestTimeoutMillis < 1 || writeTimeoutMillis < 1) {
            throw new IllegalArgumentException("Invalid HTTP resource limits");
        }
    }
}
