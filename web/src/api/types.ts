export type CardView = {
  id: string
  suit: string
  rank: string
  display: string
  played_by: number | null
  purpose: string | null
  point_value: number
  is_stolen: boolean
  attachments: CardView[]
}

export type ActionView = {
  id: number
  label: string
  type: string
  played_by: number
  source: string
  requires_additional_input: boolean
  card: CardView | null
  target: CardView | null
}

export type GameStateView = {
  hands: CardView[][]
  hand_counts: number[]
  fields: CardView[][]
  effective_fields: CardView[][]
  deck_count: number
  discard_pile: CardView[]
  discard_count: number
  scores: number[]
  targets: number[]
  turn: number
  current_action_player: number
  status: string | null
  resolving_two: boolean
  resolving_one_off: boolean
  resolving_three: boolean
  resolving_seven: boolean
  pending_seven_player: number | null
  pending_seven_cards: CardView[]
  pending_seven_requires_discard: boolean
  resolving_four: boolean
  overall_turn: number
  use_ai: boolean
  one_off_card_to_counter: CardView | null
  pending_four_count: number
}

export type SessionResponse = {
  session_id: string
  state: GameStateView
  legal_actions: ActionView[]
  state_version: number
  ai_thinking: boolean
}

export type ActionResponse = {
  state: GameStateView
  legal_actions: ActionView[]
  state_version: number
  last_actions: ActionView[]
}

export type AiType = 'llm' | 'rl'

export type HistoryEntry = {
  timestamp: string
  turn_number: number
  player: number
  action_type: string
  description: string
}

export type HistoryResponse = {
  entries: HistoryEntry[]
  turn_counter: number
}
