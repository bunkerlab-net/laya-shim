// Serves the test page and forwards requests to laya-shim. The shim sends no
// CORS headers, so the browser can't call it from another origin directly.
import page from "./index.html";

const host = process.env.LAYA_HOST ?? "127.0.0.1";
const shim = `http://${host}:${process.env.LAYA_PORT ?? "8765"}/v1/systemone`;

const server = Bun.serve({
  port: Number(process.env.UI_PORT ?? 3000),
  routes: {
    "/": page,
    "/api/systemone": {
      async POST(req) {
        try {
          const res = await fetch(shim, { method: "POST", body: await req.text() });
          return new Response(res.body, {
            status: res.status,
            headers: { "Content-Type": "application/json" },
          });
        } catch {
          return Response.json({ error: `can't reach laya-shim at ${shim}` }, { status: 502 });
        }
      },
    },
  },
});

console.log(`UI on ${server.url}, forwarding to ${shim}`);
