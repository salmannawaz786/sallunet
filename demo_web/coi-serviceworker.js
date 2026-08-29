/* Cross-origin isolation via a service worker.
 *
 * onnxruntime-web can only start its wasm thread pool when SharedArrayBuffer is
 * available, which requires COOP: same-origin + COEP: require-corp. A static
 * host (Hugging Face Spaces, GitHub Pages) cannot set those headers, so this
 * worker re-serves every same-origin response with them attached and reloads
 * the page once on first install.
 *
 * Everything the page loads is same-origin, so require-corp does not block
 * anything. If registration fails the page still works — single-threaded.
 *
 * Adapted from github.com/gzuidhof/coi-serviceworker (MIT).
 */
if (typeof window === "undefined") {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

  self.addEventListener("fetch", (event) => {
    const r = event.request;
    if (r.cache === "only-if-cached" && r.mode !== "same-origin") return;

    event.respondWith(
      fetch(r).then((response) => {
        if (response.status === 0) return response;
        const headers = new Headers(response.headers);
        headers.set("Cross-Origin-Embedder-Policy", "require-corp");
        headers.set("Cross-Origin-Opener-Policy", "same-origin");
        headers.set("Cross-Origin-Resource-Policy", "cross-origin");
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers,
        });
      }).catch((e) => console.error(e))
    );
  });
} else {
  (() => {
    if (window.crossOriginIsolated) return;
    if (!window.isSecureContext || !navigator.serviceWorker) return;

    navigator.serviceWorker
      .register(window.document.currentScript.src)
      .then((reg) => {
        reg.addEventListener("updatefound", () => window.location.reload());
        if (reg.active && !navigator.serviceWorker.controller) {
          window.location.reload();
        }
      })
      .catch((e) => console.warn("COI service worker failed:", e));
  })();
}
