import { type ChangeEvent, useEffect, useMemo, useRef, useState } from 'react'
import { Club, Diamond, Heart, Spade } from 'lucide-react'
import './App.css'
import { useGameSession } from './api/hooks'
import type { ActionView, AiType, CardView } from './api/types'

function App() {
  const [aiType, setAiType] = useState<AiType>('rl')
  const {
    session,
    history,
    isLoading,
    error,
    submitAction,
    isSubmitting,
    startNewSession,
  } = useGameSession({ aiType })
  const [selectedCardId, setSelectedCardId] = useState<string | null>(null)
  const [selectedActionId, setSelectedActionId] = useState<number | null>(null)

  const legalActions = session?.legal_actions ?? []
  const state = session?.state
  const playerHand = state?.hands[0] ?? []
  const opponentField = state?.effective_fields[1] ?? []
  const playerField = state?.effective_fields[0] ?? []
  const historyEntries = history?.entries ?? []
  const historyListRef = useRef<HTMLUListElement | null>(null)
  const lastEntryRef = useRef<HTMLLIElement | null>(null)
  const modalActions = useMemo(
    () =>
      legalActions.filter(
        (action) => action.type === 'Resolve' || action.type === 'Counter',
      ),
    [legalActions],
  )
  const discardActions = useMemo(
    () => legalActions.filter((action) => action.type === 'Take From Discard'),
    [legalActions],
  )
  const fourDiscardActions = useMemo(
    () => legalActions.filter((action) => action.type === 'Discard From Hand'),
    [legalActions],
  )
  const sevenActions = useMemo(
    () =>
      legalActions.filter((action) =>
        [
          'Points',
          'One-Off',
          'Face Card',
          'Jack',
          'Scuttle',
          'Discard Revealed',
        ].includes(action.type),
      ),
    [legalActions],
  )
  const modalActive =
    Boolean(state?.resolving_one_off) &&
    state?.current_action_player === 0 &&
    !state?.resolving_three &&
    !state?.resolving_four &&
    !state?.resolving_seven
  const discardModalActive =
    Boolean(state?.resolving_three) && state?.current_action_player === 0
  const fourDiscardModalActive =
    Boolean(state?.resolving_four) && state?.current_action_player === 0
  const sevenModalActive =
    Boolean(state?.resolving_seven) && state?.current_action_player === 0
  const sevenCards = state?.pending_seven_cards ?? []
  const sevenActionsByCard = useMemo(() => {
    const grouped = new Map<string, ActionView[]>()
    sevenActions.forEach((action) => {
      const cardId = action.card?.id
      if (!cardId) return
      const list = grouped.get(cardId) ?? []
      list.push(action)
      grouped.set(cardId, list)
    })
    return grouped
  }, [sevenActions])
  const modalCard = state?.one_off_card_to_counter ?? null

  const actionChoices = useMemo(() => {
    if (!selectedCardId) {
      return legalActions.filter((action) => action.type === 'Draw')
    }
    const matching = legalActions.filter(
      (action) => action.card?.id === selectedCardId,
    )
    return matching.length ? matching : legalActions
  }, [legalActions, selectedCardId])

  const actionSummary = useMemo(() => {
    if (!legalActions.length) return []
    return Array.from(new Set(legalActions.map((action) => action.type))).slice(
      0,
      3,
    )
  }, [legalActions])

  const groupedHistory = useMemo(() => {
    const groups: Array<{ turn: number; entries: typeof historyEntries }> = []
    historyEntries.forEach((entry) => {
      const existing = groups.find((group) => group.turn === entry.turn_number)
      if (existing) {
        existing.entries.push(entry)
      } else {
        groups.push({ turn: entry.turn_number, entries: [entry] })
      }
    })
    return groups
  }, [historyEntries])

  const selectedAction = legalActions.find(
    (action) => action.id === selectedActionId,
  )
  const isGameOver = state?.status === 'win'
  const winnerLabel = useMemo(() => {
    if (!state) return null
    if (state.scores[0] >= state.targets[0]) return 'You'
    if (state.scores[1] >= state.targets[1]) return 'AI'
    return null
  }, [state])

  useEffect(() => {
    if (discardModalActive) {
      if (selectedActionId === null && discardActions.length > 0) {
        setSelectedActionId(discardActions[0].id)
      }
      return
    }
    if (fourDiscardModalActive) {
      if (selectedActionId === null && fourDiscardActions.length > 0) {
        setSelectedActionId(fourDiscardActions[0].id)
      }
      return
    }
    if (sevenModalActive) {
      if (selectedActionId === null && sevenActions.length > 0) {
        setSelectedActionId(sevenActions[0].id)
      }
      return
    }
    if (!modalActive) return
    if (selectedActionId !== null) return
    if (modalActions.length > 0) {
      setSelectedActionId(modalActions[0].id)
    }
  }, [
    discardModalActive,
    discardActions,
    fourDiscardModalActive,
    fourDiscardActions,
    sevenModalActive,
    sevenActions,
    modalActive,
    modalActions,
    selectedActionId,
  ])

  useEffect(() => {
    if (!historyListRef.current) return
    historyListRef.current.scrollTop = historyListRef.current.scrollHeight
  }, [historyEntries.length])

  const handleActionSelect = (action: ActionView) => {
    setSelectedActionId(action.id)
  }

  const handleConfirm = () => {
    if (!session || selectedActionId === null) return
    submitAction({
      actionId: selectedActionId,
      stateVersion: session.state_version,
    })
    setSelectedActionId(null)
    setSelectedCardId(null)
  }

  const handleAiChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const nextType = event.target.value as AiType
    if (nextType === aiType) return
    setAiType(nextType)
    startNewSession(nextType)
  }

  return (
    <div className="app">
      <header className="top-bar">
        <div className="brand">
          <span className="brand-mark">Cuttle</span>
          <span className="brand-sub">AI sparring table</span>
        </div>
        <div className="scoreboard">
          <div className="score-chip">
            <span className="chip-label">You</span>
            <span className="chip-score">
              {state ? `${state.scores[0]} / ${state.targets[0]}` : '--'}
            </span>
          </div>
          <div className="score-chip muted">
            <span className="chip-label">AI</span>
            <span className="chip-score">
              {state ? `${state.scores[1]} / ${state.targets[1]}` : '--'}
            </span>
          </div>
        </div>
        <div className="controls">
          <label className="ai-select">
            <span>AI</span>
            <select value={aiType} onChange={handleAiChange}>
              <option value="rl">RL</option>
              <option value="llm">LLM</option>
            </select>
          </label>
          <button className="ghost">History</button>
          <button className="ghost" onClick={() => startNewSession()}>
            New Game
          </button>
          <button className="primary" onClick={() => startNewSession()}>
            End Game
          </button>
        </div>
      </header>

      <main className="table">
        <aside className="rail left">
          <div className="stack-card deck">
            <span className="stack-title">Deck</span>
            <span className="stack-count">{state?.deck_count ?? '--'}</span>
          </div>
          <div className="stack-card scrap">
            <span className="stack-title">Scrap</span>
            <span className="stack-count">{state?.discard_count ?? '--'}</span>
            <button className="ghost small">View</button>
          </div>
        </aside>

        <section className="table-surface">
          <div className="zone opponent">
            <div className="zone-label-row">
              <div className="zone-label">AI Field</div>
              <div className="zone-chip">
                AI hand: {state?.hand_counts?.[1] ?? '--'}
              </div>
            </div>
            <div className="card-row">
              {opponentField.map((card) => (
                <CardTile key={card.id} card={card} />
              ))}
            </div>
          </div>

          <div className="center-zone">
            <div className="phase-pill">
              {state?.current_action_player === 0 ? 'Your turn' : 'AI thinking'} ·
              {state?.status ?? 'Choose an action'}
            </div>
            <div className="action-hints">
              {actionSummary.map((action) => (
                <span
                  key={action}
                  className={`hint ${action === 'Points' ? 'accent' : ''}`}
                >
                  {action}
                </span>
              ))}
            </div>
          </div>

          <div className="zone player">
            <div className="zone-label">Your Field</div>
            <div className="card-row">
              {playerField.map((card) => (
                <CardTile key={card.id} card={card} />
              ))}
            </div>
          </div>
        </section>

        <aside className="rail right">
          <div className="history-panel">
            <div className="panel-header">
              <span>History</span>
              <button
                className="ghost small"
                onClick={() => {
                  if (historyListRef.current) {
                    historyListRef.current.scrollTo({
                      top: historyListRef.current.scrollHeight,
                      behavior: 'auto',
                    })
                  } else {
                    lastEntryRef.current?.scrollIntoView({
                      behavior: 'auto',
                      block: 'end',
                    })
                  }
                }}
              >
                Jump to last
              </button>
            </div>
            <ul ref={historyListRef}>
              {groupedHistory.map((group) => (
                <li key={`turn-${group.turn}`} className="history-turn">
                  Turn {group.turn}
                  <ul>
                    {group.entries.map((entry, index) => {
                      const isLast =
                        entry === historyEntries[historyEntries.length - 1]
                      return (
                        <li
                          key={`${entry.timestamp}-${entry.description}-${index}`}
                          className="history-entry"
                          ref={isLast ? lastEntryRef : null}
                        >
                          {entry.description}
                        </li>
                      )
                    })}
                  </ul>
                </li>
              ))}
            </ul>
          </div>
        </aside>
      </main>

      <footer className="hand-area">
        <div className="action-prompt">
          <div>
            <div className="action-title">
              {selectedAction?.label ?? 'Select an action'}
            </div>
            {selectedAction && actionDescription(selectedAction) ? (
              <div className="action-desc-line">
                {actionDescription(selectedAction)}
              </div>
            ) : null}
            <div className="action-sub">
              {state?.resolving_one_off
                ? 'Resolve or counter the one-off.'
                : 'Choose a card and action to continue.'}
            </div>
          </div>
          <div className="action-buttons">
            <button
              className="ghost"
              onClick={() => setSelectedActionId(null)}
              disabled={!selectedAction}
            >
              Cancel
            </button>
            <button
              className="primary"
              onClick={handleConfirm}
              disabled={
                selectedActionId === null ||
                isSubmitting ||
                !session ||
                Boolean(isGameOver)
              }
            >
              {isSubmitting ? 'Sending...' : 'Confirm'}
            </button>
          </div>
        </div>
        <div className="action-choice-row">
          {actionChoices.map((action) => (
            <ActionButton
              key={action.id}
              action={action}
              context="choice"
              selected={selectedActionId === action.id}
              disabled={Boolean(isGameOver)}
              onClick={() => handleActionSelect(action)}
            />
          ))}
        </div>
        <div className="hand">
          {playerHand.map((card) => (
            <CardTile
              key={card.id}
              card={card}
              isHand
              selected={selectedCardId === card.id}
              onClick={() =>
                setSelectedCardId((prev) => (prev === card.id ? null : card.id))
              }
            />
          ))}
        </div>
      </footer>
      {isGameOver && (
        <div className="modal-scrim">
          <div className="modal">
            <div className="modal-header">
              <div>
                <div className="modal-title">Game Over</div>
                <div className="modal-sub">
                  {winnerLabel ? `${winnerLabel} win.` : 'Game ended.'}
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="primary" onClick={() => startNewSession()}>
                New Game
              </button>
            </div>
          </div>
        </div>
      )}
      {discardModalActive && (
        <div className="modal-scrim">
          <div className="modal">
            <div className="modal-header">
              <div>
                <div className="modal-title">Take From Scrap</div>
                <div className="modal-sub">
                  Choose one card from the scrap pile to add to your hand.
                </div>
              </div>
              <button
                className="ghost small"
                onClick={() => setSelectedActionId(null)}
              >
                Close
              </button>
            </div>
            <div className="modal-body">
              <div className="modal-actions">
                {discardActions.map((action) => (
                  <button
                    key={action.id}
                    className={`ghost ${
                      selectedActionId === action.id ? 'selected-action' : ''
                    }`}
                    onClick={() => handleActionSelect(action)}
                  >
                    {action.card ? action.card.display : action.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="modal-footer">
              <button
                className="primary"
                onClick={handleConfirm}
                disabled={selectedActionId === null || isSubmitting || !session}
              >
                {isSubmitting ? 'Sending...' : 'Take card'}
              </button>
            </div>
          </div>
        </div>
      )}
      {fourDiscardModalActive && (
        <div className="modal-scrim">
          <div className="modal">
            <div className="modal-header">
              <div>
                <div className="modal-title">Discard Cards</div>
                <div className="modal-sub">
                  Select {state?.pending_four_count ?? 0} card
                  {state?.pending_four_count === 1 ? '' : 's'} to discard.
                </div>
              </div>
              <button
                className="ghost small"
                onClick={() => setSelectedActionId(null)}
              >
                Close
              </button>
            </div>
            <div className="modal-body">
              <div className="modal-actions">
                {fourDiscardActions.map((action) => (
                  <button
                    key={action.id}
                    className={`ghost ${
                      selectedActionId === action.id ? 'selected-action' : ''
                    }`}
                    onClick={() => handleActionSelect(action)}
                  >
                    {action.card ? action.card.display : action.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="modal-footer">
              <button
                className="primary"
                onClick={handleConfirm}
                disabled={selectedActionId === null || isSubmitting || !session}
              >
                {isSubmitting ? 'Sending...' : 'Discard'}
              </button>
            </div>
          </div>
        </div>
      )}
      {sevenModalActive && (
        <div className="modal-scrim">
          <div className="modal">
            <div className="modal-header">
              <div>
                <div className="modal-title">Seven Reveal</div>
                <div className="modal-sub">
                  Choose how to play one revealed card.
                </div>
              </div>
              <button
                className="ghost small"
                onClick={() => setSelectedActionId(null)}
              >
                Close
              </button>
            </div>
            <div className="modal-body">
              <div className="seven-card-grid">
                {sevenCards.map((card) => (
                  <div key={card.id} className="seven-card">
                    <CardTile card={card} />
                    <div className="modal-actions">
                      {(sevenActionsByCard.get(card.id) ?? []).map((action) => (
                        <ActionButton
                          key={action.id}
                          action={action}
                          context="seven"
                          selected={selectedActionId === action.id}
                          onClick={() => handleActionSelect(action)}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="modal-footer">
              <button
                className="primary"
                onClick={handleConfirm}
                disabled={selectedActionId === null || isSubmitting || !session}
              >
                {isSubmitting ? 'Sending...' : 'Confirm'}
              </button>
            </div>
          </div>
        </div>
      )}
      {modalActive && !discardModalActive && (
        <div className="modal-scrim">
          <div className="modal">
            <div className="modal-header">
              <div>
                <div className="modal-title">Resolve One-Off</div>
                <div className="modal-sub">
                  Choose to counter or resolve the pending action.
                </div>
              </div>
              <button
                className="ghost small"
                onClick={() => setSelectedActionId(null)}
              >
                Close
              </button>
            </div>
            <div className="modal-body">
              <div className="modal-card">
                {modalCard ? (
                  <CardTile card={modalCard} />
                ) : (
                  <div className="modal-card-placeholder">No target</div>
                )}
              </div>
              <div className="modal-actions">
                {modalActions.map((action) => (
                  <ActionButton
                    key={action.id}
                    action={action}
                    context="resolve"
                    selected={selectedActionId === action.id}
                    onClick={() => handleActionSelect(action)}
                  />
                ))}
              </div>
            </div>
            <div className="modal-footer">
              <button
                className="primary"
                onClick={handleConfirm}
                disabled={selectedActionId === null || isSubmitting || !session}
              >
                {isSubmitting ? 'Sending...' : 'Confirm'}
              </button>
            </div>
          </div>
        </div>
      )}
      {isLoading && <div className="status-banner">Starting session...</div>}
      {error && (
        <div className="status-banner error">
          Failed to load session. Check server.
        </div>
      )}
    </div>
  )
}

type CardTileProps = {
  card: CardView
  selected?: boolean
  isHand?: boolean
  onClick?: () => void
}

type ActionButtonProps = {
  action: ActionView
  context: string
  selected?: boolean
  disabled?: boolean
  onClick: () => void
}

const oneOffRanks = new Set(['ACE', 'THREE', 'FOUR', 'FIVE', 'SIX'])
const faceRanks = new Set(['JACK', 'QUEEN', 'KING', 'EIGHT'])
const oneOffDescriptions: Record<string, string> = {
  ACE: 'Destroy all point cards on the field.',
  TWO: 'Scrap a target royal or Glasses Eight.',
  THREE: 'Take one card from the scrap pile.',
  FOUR: 'Opponent discards two cards of their choice.',
  FIVE: 'Discard one card, then draw up to three (max 8 in hand).',
  SIX: 'Destroy all face cards (royals) on the field.',
  SEVEN: 'Reveal the top two cards of the deck; choose one to play.',
  NINE: "Return an opponent's field card to their hand (cannot play it next turn).",
  TEN: 'No effect.',
}
const faceCardDescriptions: Record<string, string> = {
  KING: 'Reduce points needed to win while in play.',
  QUEEN: 'Protect your other cards from targeted effects.',
  JACK: 'Steal a target point card while in play.',
  EIGHT: "Reveal your opponent's hand while in play.",
}

function cardTag(card: CardView) {
  if (card.purpose === 'POINTS') return 'points'
  if (card.purpose === 'FACE_CARD' || card.purpose === 'JACK') return 'face'
  if (card.purpose === 'ONE_OFF' || card.purpose === 'COUNTER') return 'one-off'
  if (oneOffRanks.has(card.rank)) return 'one-off'
  if (faceRanks.has(card.rank)) return 'face'
  return 'points'
}

function rankShort(rank: string) {
  const map: Record<string, string> = {
    ACE: 'A',
    TWO: '2',
    THREE: '3',
    FOUR: '4',
    FIVE: '5',
    SIX: '6',
    SEVEN: '7',
    EIGHT: '8',
    NINE: '9',
    TEN: '10',
    JACK: 'J',
    QUEEN: 'Q',
    KING: 'K',
  }
  return map[rank] ?? rank
}

function actionDescription(action: ActionView): string | null {
  if (action.type === 'Draw') {
    return 'Draw one card (max 8 in hand).'
  }
  if (action.type === 'Points') {
    return 'Play for points equal to its rank.'
  }
  if (action.type === 'Scuttle') {
    return 'Scrap an opponent point card with a higher card (suit breaks ties).'
  }
  if (action.type === 'Face Card') {
    if (action.card?.rank && faceCardDescriptions[action.card.rank]) {
      return faceCardDescriptions[action.card.rank]
    }
    return 'Play a royal for an ongoing effect.'
  }
  if (action.type === 'Jack') {
    return 'Steal a target point card while in play.'
  }
  if (action.type === 'One-Off') {
    if (action.card?.rank && oneOffDescriptions[action.card.rank]) {
      return oneOffDescriptions[action.card.rank]
    }
    return "Trigger this card's one-off effect."
  }
  if (action.type === 'Counter') {
    return 'Counter a one-off with a Two.'
  }
  if (action.type === 'Resolve') {
    return 'Let the pending one-off resolve.'
  }
  if (action.type === 'Take From Discard') {
    return 'Take one card from the scrap pile.'
  }
  if (action.type === 'Discard From Hand') {
    return 'Discard a card from your hand.'
  }
  if (action.type === 'Discard Revealed') {
    return 'Discard a revealed card.'
  }
  if (action.type === 'Request Stalemate') {
    return 'Ask to end the game in a stalemate.'
  }
  if (action.type === 'Accept Stalemate') {
    return 'Accept the stalemate request.'
  }
  if (action.type === 'Reject Stalemate') {
    return 'Reject the stalemate request.'
  }
  if (action.type === 'Concede') {
    return 'Concede the game.'
  }
  return null
}

function SuitIcon({ suit, size = 20 }: { suit: string; size?: number }) {
  const suitProps = { size, strokeWidth: 2.5 }
  
  switch (suit) {
    case 'CLUBS':
      return <Club {...suitProps} className="suit-icon suit-black" />
    case 'DIAMONDS':
      return <Diamond {...suitProps} className="suit-icon suit-red" />
    case 'HEARTS':
      return <Heart {...suitProps} className="suit-icon suit-red" />
    case 'SPADES':
      return <Spade {...suitProps} className="suit-icon suit-black" />
    default:
      return <span>{suit}</span>
  }
}

function ActionButton({
  action,
  context,
  selected,
  disabled,
  onClick,
}: ActionButtonProps) {
  const description = actionDescription(action)
  const descId = description ? `action-desc-${context}-${action.id}` : undefined

  return (
    <button
      className={`ghost action-button ${selected ? 'selected-action' : ''}`}
      onClick={onClick}
      disabled={disabled}
      aria-label={action.label}
      aria-describedby={descId}
    >
      <span className="action-label">{action.label}</span>
      {description ? (
        <span className="action-desc" id={descId}>
          {description}
        </span>
      ) : null}
    </button>
  )
}

function CardTile({ card, selected, isHand, onClick }: CardTileProps) {
  const tag = cardTag(card)
  const rank = rankShort(card.rank)

  return (
    <div
      className={[
        'card-tile',
        `card-${tag}`,
        selected ? 'selected' : '',
        isHand ? 'hand-card' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      onClick={onClick}
      data-testid={`card-${card.id}`}
    >
      <div className="card-rank">{rank}</div>
      <div className="card-suit">
        <SuitIcon suit={card.suit} />
      </div>
      <div className="card-tag">{tag}</div>
    </div>
  )
}

export default App
