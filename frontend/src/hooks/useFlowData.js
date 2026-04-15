/**
 * useFlowData.js
 *
 * Custom React hook. This is the bridge between the raw WebSocket
 * and your React components.
 *
 * What is a custom hook?
 * It's just a function that starts with "use" and can call other hooks
 * like useState and useEffect. It lets you share stateful logic between
 * components without repeating code.
 *
 * What does this hook return?
 *   data      — the latest payload from ai_modules (or null during warmup)
 *   connected — string: 'connected' | 'disconnected' | 'error'
 *   history   — last 60 pulse samples for the live waveform chart
 *
 * Any component that calls useFlowData() gets live-updating data
 * automatically. When a new payload arrives, React re-renders the
 * component with fresh values. You never manually "refresh" anything.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { flowSocket } from '../services/websocket';

const MAX_HISTORY = 60;   // keep 60 seconds of pulse samples

export function useFlowData() {
  const [data,      setData]      = useState(null);
  const [connected, setConnected] = useState('disconnected');
  const [history,   setHistory]   = useState([]);   // pulse waveform samples
  const mountedRef  = useRef(true);

  const handleData = useCallback((payload) => {
    if (!mountedRef.current) return;
    setData(payload);

    // Accumulate pulse samples for waveform chart
    const sample = payload?.vitals?.pulse_sample ?? 0;
    setHistory(prev => {
      const next = [...prev, sample];
      return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
    });
  }, []);

  const handleStatus = useCallback((status) => {
    if (!mountedRef.current) return;
    setConnected(status);
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    flowSocket.connect(handleData, handleStatus);

    return () => {
      mountedRef.current = false;
      flowSocket.disconnect();
    };
  }, [handleData, handleStatus]);

  return { data, connected, history };
}