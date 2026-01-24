import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import {
  createSession,
  fetchHistory,
  fetchSession,
  submitAction,
} from './client'
import type {
  ActionResponse,
  AiType,
  HistoryResponse,
  SessionResponse,
} from './types'

type ActionArgs = { actionId: number; stateVersion: number }

type Options = { autoStart?: boolean; aiType?: AiType }

export function useGameSession(options: Options = {}) {
  const queryClient = useQueryClient()
  const [sessionId, setSessionId] = useState<string | null>(null)
  const hasStartedRef = useRef(false)

  const createSessionMutation = useMutation<SessionResponse, Error, AiType>({
    mutationFn: (aiType: AiType) => createSession(aiType),
    onSuccess: (data) => {
      setSessionId(data.session_id)
      queryClient.setQueryData(['session', data.session_id], data)
    },
  })

  const sessionQuery = useQuery<SessionResponse, Error>({
    queryKey: ['session', sessionId],
    queryFn: () => fetchSession(sessionId as string),
    enabled: Boolean(sessionId),
    refetchInterval: (query) =>
      query.state.data?.ai_thinking ? 1500 : false,
  })

  const historyQuery = useQuery<HistoryResponse, Error>({
    queryKey: ['history', sessionId],
    queryFn: () => fetchHistory(sessionId as string),
    enabled: Boolean(sessionId),
  })

  const actionMutation = useMutation<ActionResponse, Error, ActionArgs>({
    mutationFn: ({ actionId, stateVersion }: ActionArgs) =>
      submitAction(sessionId as string, stateVersion, actionId),
    onSuccess: (data: ActionResponse) => {
      queryClient.setQueryData(
        ['session', sessionId],
        (prev?: SessionResponse) =>
          prev
            ? {
                ...prev,
                state: data.state,
                legal_actions: data.legal_actions,
                state_version: data.state_version,
                ai_thinking: false,
              }
            : prev,
      )
      queryClient.invalidateQueries({ queryKey: ['history', sessionId] })
    },
  })

  const startNewSession = (nextAiType?: AiType) => {
    const resolvedAiType =
      nextAiType === 'llm' || nextAiType === 'rl'
        ? nextAiType
        : options.aiType ?? 'rl'
    const previousSession = sessionId
    if (previousSession) {
      queryClient.removeQueries({ queryKey: ['session', previousSession] })
      queryClient.removeQueries({ queryKey: ['history', previousSession] })
    }
    setSessionId(null)
    createSessionMutation.mutate(resolvedAiType)
  }

  useEffect(() => {
    if (options.autoStart === false) return
    if (!sessionId && !createSessionMutation.isPending && !hasStartedRef.current) {
      hasStartedRef.current = true
      createSessionMutation.mutate(options.aiType ?? 'rl')
    }
  }, [options.aiType, options.autoStart, sessionId, createSessionMutation])

  const error = createSessionMutation.error ?? sessionQuery.error ?? null
  const sessionData = sessionQuery.data ?? createSessionMutation.data ?? null
  const isLoading =
    createSessionMutation.isPending || (!sessionData && !error && !sessionId)

  return {
    sessionId,
    session: sessionData,
    history: historyQuery.data ?? null,
    isLoading,
    error,
    submitAction: actionMutation.mutate,
    isSubmitting: actionMutation.isPending,
    startNewSession,
  }
}
