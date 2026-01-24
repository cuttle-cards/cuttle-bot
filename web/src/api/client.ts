import type {
  ActionResponse,
  AiType,
  HistoryResponse,
  SessionResponse,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })

  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || response.statusText)
  }

  return (await response.json()) as T
}

export function createSession(aiType: AiType = 'rl'): Promise<SessionResponse> {
  return request('/api/sessions', {
    method: 'POST',
    body: JSON.stringify({
      use_ai: true,
      manual_selection: false,
      ai_type: aiType,
    }),
  })
}

export function fetchSession(sessionId: string): Promise<SessionResponse> {
  return request(`/api/sessions/${sessionId}`)
}

export function submitAction(
  sessionId: string,
  stateVersion: number,
  actionId: number,
): Promise<ActionResponse> {
  return request(`/api/sessions/${sessionId}/actions`, {
    method: 'POST',
    body: JSON.stringify({ state_version: stateVersion, action_id: actionId }),
  })
}

export function fetchHistory(sessionId: string): Promise<HistoryResponse> {
  return request(`/api/sessions/${sessionId}/history`)
}
