import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { vi } from 'vitest'

import { useGameSession } from '../../src/api/hooks'

const mockSession = {
  session_id: 'test-session',
  state: {
    hands: [[], []],
    hand_counts: [0, 0],
    fields: [[], []],
    effective_fields: [[], []],
    deck_count: 20,
    discard_pile: [],
    discard_count: 0,
    scores: [0, 0],
    targets: [21, 21],
    turn: 0,
    current_action_player: 0,
    status: null,
    resolving_two: false,
    resolving_one_off: false,
    resolving_three: false,
    overall_turn: 0,
    use_ai: true,
    one_off_card_to_counter: null,
  },
  legal_actions: [],
  state_version: 0,
  ai_thinking: false,
}

const mockHistory = {
  entries: [],
  turn_counter: 0,
}

function Wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

function HookConsumer() {
  const { session, isLoading } = useGameSession()
  if (isLoading) {
    return <div>loading</div>
  }
  return <div>{session?.session_id ?? 'none'}</div>
}

test('creates a session and exposes session id', async () => {
  const fetchMock = vi
    .fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify(mockSession), { status: 200 }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify(mockHistory), { status: 200 }),
    )

  vi.stubGlobal('fetch', fetchMock)

  render(<HookConsumer />, { wrapper: Wrapper })

  await waitFor(() =>
    expect(screen.getByText('test-session')).toBeInTheDocument(),
  )

  vi.unstubAllGlobals()
})
