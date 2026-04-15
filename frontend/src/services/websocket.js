/**
 * websocket.js
 *
 * Raw WebSocket manager. This is the ONLY file that talks to the backend.
 * It is not a React component — it's a plain JS class.
 *
 * Why a class and not just a hook?
 * Because we want one persistent connection that survives re-renders.
 * The hook (useFlowData.js) wraps this and makes it React-friendly.
 */

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws/frontend';
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS  = 30000;

class FlowWebSocket {
  constructor() {
    this.ws            = null;
    this.onData        = null;   // callback(payload)
    this.onStatusChange= null;   // callback('connected' | 'disconnected' | 'error')
    this._backoff      = RECONNECT_BASE_MS;
    this._destroyed    = false;
    this._retryTimer   = null;
  }

  connect(onData, onStatusChange) {
    this.onData         = onData;
    this.onStatusChange = onStatusChange;
    this._destroyed     = false;
    this._open();
  }

  _open() {
    if (this._destroyed) return;

    try {
      this.ws = new WebSocket(WS_URL);
    } catch (e) {
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this._backoff = RECONNECT_BASE_MS;   // reset on success
      this.onStatusChange?.('connected');
    };

    this.ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        this.onData?.(payload);
      } catch (e) {
        console.warn('[WS] Could not parse message:', e);
      }
    };

    this.ws.onerror = () => {
      this.onStatusChange?.('error');
    };

    this.ws.onclose = () => {
      this.onStatusChange?.('disconnected');
      this._scheduleReconnect();
    };
  }

  _scheduleReconnect() {
    if (this._destroyed) return;
    clearTimeout(this._retryTimer);
    this._retryTimer = setTimeout(() => {
      this._backoff = Math.min(this._backoff * 2, RECONNECT_MAX_MS);
      this._open();
    }, this._backoff);
  }

  disconnect() {
    this._destroyed = true;
    clearTimeout(this._retryTimer);
    if (this.ws) {
      this.ws.onclose = null;   // prevent reconnect loop on intentional close
      this.ws.close();
      this.ws = null;
    }
  }
}

// Singleton — one connection for the whole app
export const flowSocket = new FlowWebSocket();