import { expect, test } from '@playwright/test'

test('renders core table layout', async ({ page }) => {
  await page.route('**/api/sessions', async (route) => {
    const payload = {
      session_id: 'test-session',
      state: {
        hands: [
          [
            {
              id: 'card-1',
              suit: 'SPADES',
              rank: 'ACE',
              display: 'Ace of Spades',
              played_by: null,
              purpose: null,
              point_value: 1,
              is_stolen: false,
              attachments: [],
            },
          ],
          [],
        ],
        hand_counts: [1, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 19,
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
      legal_actions: [
        {
          id: 0,
          label: 'Play Ace of Spades as points',
          type: 'Points',
          played_by: 0,
          source: 'Hand',
          requires_additional_input: false,
          card: {
            id: 'card-1',
            suit: 'SPADES',
            rank: 'ACE',
            display: 'Ace of Spades',
            played_by: null,
            purpose: null,
            point_value: 1,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
        {
          id: 1,
          label: 'Draw a card from deck',
          type: 'Draw',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: null,
          target: null,
        },
      ],
      state_version: 0,
      ai_thinking: false,
    }

    await route.fulfill({ json: payload })
  })

  await page.route('**/api/sessions/test-session/history', async (route) => {
    const entries = Array.from({ length: 12 }, (_, index) => ({
      timestamp: `2025-01-01T00:00:${index.toString().padStart(2, '0')}Z`,
      turn_number: index < 6 ? 1 : 2,
      player: index % 2,
      action_type: 'Draw',
      description: `History entry ${index + 1}`,
    }))

    await route.fulfill({ json: { entries, turn_counter: 2 } })
  })

  await page.route('**/api/sessions/test-session', async (route) => {
    const payload = {
      session_id: 'test-session',
      state: {
        hands: [
          [
            {
              id: 'card-1',
              suit: 'SPADES',
              rank: 'ACE',
              display: 'Ace of Spades',
              played_by: null,
              purpose: null,
              point_value: 1,
              is_stolen: false,
              attachments: [],
            },
          ],
          [],
        ],
        hand_counts: [1, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 19,
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
      legal_actions: [
        {
          id: 0,
          label: 'Play Ace of Spades as points',
          type: 'Points',
          played_by: 0,
          source: 'Hand',
          requires_additional_input: false,
          card: {
            id: 'card-1',
            suit: 'SPADES',
            rank: 'ACE',
            display: 'Ace of Spades',
            played_by: null,
            purpose: null,
            point_value: 1,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
        {
          id: 1,
          label: 'Draw a card from deck',
          type: 'Draw',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: null,
          target: null,
        },
      ],
      state_version: 0,
      ai_thinking: false,
    }

    await route.fulfill({ json: payload })
  })

  await page.route('**/api/sessions/test-session/actions', async (route) => {
    const payload = {
      state: {
        hands: [[], []],
        hand_counts: [0, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 18,
        discard_pile: [],
        discard_count: 0,
        scores: [1, 0],
        targets: [21, 21],
        turn: 1,
        current_action_player: 1,
        status: null,
        resolving_two: false,
        resolving_one_off: false,
        resolving_three: false,
        overall_turn: 0,
        use_ai: true,
        one_off_card_to_counter: null,
      },
      legal_actions: [],
      state_version: 1,
      last_actions: [
        {
          id: -1,
          label: 'Play Ace of Spades as points',
          type: 'Points',
          played_by: 0,
          source: 'Hand',
          requires_additional_input: false,
          card: null,
          target: null,
        },
      ],
    }

    await route.fulfill({ json: payload })
  })

  await page.goto('/')

  await expect(page.getByText('Cuttle')).toBeVisible()
  await expect(page.getByRole('button', { name: 'History' })).toBeVisible()
  await expect(page.getByText('Deck', { exact: true })).toBeVisible()
  await expect(page.getByText('Scrap')).toBeVisible()
  await expect(page.getByText('Your Field')).toBeVisible()

  const historyList = page.locator('.history-panel ul').first()
  const initialScroll = await historyList.evaluate((node) => node.scrollTop)
  await page.getByRole('button', { name: 'Jump to last' }).click()
  await page.waitForTimeout(200)
  const updatedScroll = await historyList.evaluate((node) => node.scrollTop)
  expect(updatedScroll).toBeGreaterThanOrEqual(initialScroll)

  await page.getByTestId('card-card-1').click()
  await page.getByRole('button', { name: 'Play Ace of Spades as points' }).click()
  const responsePromise = page.waitForResponse('**/api/sessions/test-session/actions')
  await page.getByRole('button', { name: 'Confirm' }).click()
  await responsePromise
})

test('visual snapshot', async ({ page }) => {
  await page.route('**/api/sessions', async (route) => {
    const payload = {
      session_id: 'snapshot-session',
      state: {
        hands: [
          [
            {
              id: 'card-1',
              suit: 'SPADES',
              rank: 'ACE',
              display: 'Ace of Spades',
              played_by: null,
              purpose: null,
              point_value: 1,
              is_stolen: false,
              attachments: [],
            },
          ],
          [],
        ],
        hand_counts: [1, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 19,
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

    await route.fulfill({ json: payload })
  })

  await page.route('**/api/sessions/snapshot-session/history', async (route) => {
    await route.fulfill({ json: { entries: [], turn_counter: 0 } })
  })

  await page.route('**/api/sessions/snapshot-session', async (route) => {
    const payload = {
      session_id: 'snapshot-session',
      state: {
        hands: [
          [
            {
              id: 'card-1',
              suit: 'SPADES',
              rank: 'ACE',
              display: 'Ace of Spades',
              played_by: null,
              purpose: null,
              point_value: 1,
              is_stolen: false,
              attachments: [],
            },
          ],
          [],
        ],
        hand_counts: [1, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 19,
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

    await route.fulfill({ json: payload })
  })

  await page.goto('/')
  await expect(page.getByText('Cuttle')).toBeVisible()
  await page.waitForTimeout(300)
  expect(await page.screenshot()).toMatchSnapshot('table-layout.png', {
    maxDiffPixelRatio: 0.02,
  })
})

test('mobile layout fits in one viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.emulateMedia({ reducedMotion: 'reduce' })

  await page.route('**/api/sessions', async (route) => {
    const payload = {
      session_id: 'mobile-session',
      state: {
        hands: [
          [
            {
              id: 'card-1',
              suit: 'SPADES',
              rank: 'ACE',
              display: 'Ace of Spades',
              played_by: null,
              purpose: null,
              point_value: 1,
              is_stolen: false,
              attachments: [],
            },
          ],
          [],
        ],
        hand_counts: [1, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 19,
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
      legal_actions: [
        {
          id: 0,
          label: 'Play Ace of Spades as points',
          type: 'Points',
          played_by: 0,
          source: 'Hand',
          requires_additional_input: false,
          card: {
            id: 'card-1',
            suit: 'SPADES',
            rank: 'ACE',
            display: 'Ace of Spades',
            played_by: null,
            purpose: null,
            point_value: 1,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
        {
          id: 1,
          label: 'Draw a card from deck',
          type: 'Draw',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: null,
          target: null,
        },
      ],
      state_version: 0,
      ai_thinking: false,
    }

    await route.fulfill({ json: payload })
  })

  await page.route('**/api/sessions/mobile-session/history', async (route) => {
    const entries = Array.from({ length: 6 }, (_, index) => ({
      timestamp: `2025-01-01T00:00:${index.toString().padStart(2, '0')}Z`,
      turn_number: 1,
      player: index % 2,
      action_type: 'Draw',
      description: `History entry ${index + 1}`,
    }))

    await route.fulfill({ json: { entries, turn_counter: 1 } })
  })

  await page.route('**/api/sessions/mobile-session', async (route) => {
    const payload = {
      session_id: 'mobile-session',
      state: {
        hands: [
          [
            {
              id: 'card-1',
              suit: 'SPADES',
              rank: 'ACE',
              display: 'Ace of Spades',
              played_by: null,
              purpose: null,
              point_value: 1,
              is_stolen: false,
              attachments: [],
            },
          ],
          [],
        ],
        hand_counts: [1, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 19,
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
      legal_actions: [
        {
          id: 0,
          label: 'Play Ace of Spades as points',
          type: 'Points',
          played_by: 0,
          source: 'Hand',
          requires_additional_input: false,
          card: {
            id: 'card-1',
            suit: 'SPADES',
            rank: 'ACE',
            display: 'Ace of Spades',
            played_by: null,
            purpose: null,
            point_value: 1,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
        {
          id: 1,
          label: 'Draw a card from deck',
          type: 'Draw',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: null,
          target: null,
        },
      ],
      state_version: 0,
      ai_thinking: false,
    }

    await route.fulfill({ json: payload })
  })

  await page.goto('/')
  await expect(page.getByText('Cuttle')).toBeVisible()

  const viewportHeight = await page.evaluate(() => window.innerHeight)

  const sections = [
    page.locator('.top-bar'),
    page.locator('.rail.left'),
    page.locator('.table-surface'),
    page.locator('.rail.right'),
    page.locator('.hand-area'),
  ]

  for (const section of sections) {
    const box = await section.boundingBox()
    expect(box).not.toBeNull()
    if (!box) continue
    expect(box.y).toBeGreaterThanOrEqual(0)
    expect(box.y + box.height).toBeLessThanOrEqual(viewportHeight + 2)
  }

  const fitsViewport = await page.evaluate(
    () => document.documentElement.scrollHeight <= window.innerHeight + 2,
  )
  expect(fitsViewport).toBeTruthy()

  expect(await page.screenshot()).toMatchSnapshot('table-layout-mobile.png', {
    maxDiffPixelRatio: 0.02,
  })
})

test('one-off modal flow', async ({ page }) => {
  await page.route('**/api/sessions', async (route) => {
    const payload = {
      session_id: 'modal-session',
      state: {
        hands: [[], []],
        hand_counts: [0, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 15,
        discard_pile: [],
        discard_count: 0,
        scores: [0, 0],
        targets: [21, 21],
        turn: 0,
        current_action_player: 0,
        status: null,
        resolving_two: false,
        resolving_one_off: true,
        resolving_three: false,
        overall_turn: 1,
        use_ai: true,
        one_off_card_to_counter: {
          id: 'target-1',
          suit: 'HEARTS',
          rank: 'FIVE',
          display: 'Five of Hearts',
          played_by: 1,
          purpose: 'ONE_OFF',
          point_value: 5,
          is_stolen: false,
          attachments: [],
        },
      },
      legal_actions: [
        {
          id: 0,
          label: 'Resolve one-off Five of Hearts',
          type: 'Resolve',
          played_by: 0,
          source: 'Hand',
          requires_additional_input: false,
          card: null,
          target: {
            id: 'target-1',
            suit: 'HEARTS',
            rank: 'FIVE',
            display: 'Five of Hearts',
            played_by: 1,
            purpose: 'ONE_OFF',
            point_value: 5,
            is_stolen: false,
            attachments: [],
          },
        },
      ],
      state_version: 0,
      ai_thinking: false,
    }

    await route.fulfill({ json: payload })
  })

  await page.route('**/api/sessions/modal-session/history', async (route) => {
    await route.fulfill({ json: { entries: [], turn_counter: 1 } })
  })

  await page.route('**/api/sessions/modal-session', async (route) => {
    const payload = {
      session_id: 'modal-session',
      state: {
        hands: [[], []],
        hand_counts: [0, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 15,
        discard_pile: [],
        discard_count: 0,
        scores: [0, 0],
        targets: [21, 21],
        turn: 0,
        current_action_player: 0,
        status: null,
        resolving_two: false,
        resolving_one_off: true,
        resolving_three: false,
        overall_turn: 1,
        use_ai: true,
        one_off_card_to_counter: {
          id: 'target-1',
          suit: 'HEARTS',
          rank: 'FIVE',
          display: 'Five of Hearts',
          played_by: 1,
          purpose: 'ONE_OFF',
          point_value: 5,
          is_stolen: false,
          attachments: [],
        },
      },
      legal_actions: [
        {
          id: 0,
          label: 'Resolve one-off Five of Hearts',
          type: 'Resolve',
          played_by: 0,
          source: 'Hand',
          requires_additional_input: false,
          card: null,
          target: {
            id: 'target-1',
            suit: 'HEARTS',
            rank: 'FIVE',
            display: 'Five of Hearts',
            played_by: 1,
            purpose: 'ONE_OFF',
            point_value: 5,
            is_stolen: false,
            attachments: [],
          },
        },
      ],
      state_version: 0,
      ai_thinking: false,
    }

    await route.fulfill({ json: payload })
  })

  await page.route('**/api/sessions/modal-session/actions', async (route) => {
    const payload = {
      state: {
        hands: [[], []],
        hand_counts: [0, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 15,
        discard_pile: [],
        discard_count: 0,
        scores: [0, 0],
        targets: [21, 21],
        turn: 1,
        current_action_player: 1,
        status: null,
        resolving_two: false,
        resolving_one_off: false,
        resolving_three: false,
        overall_turn: 1,
        use_ai: true,
        one_off_card_to_counter: null,
      },
      legal_actions: [],
      state_version: 1,
      last_actions: [
        {
          id: -1,
          label: 'Resolve one-off Five of Hearts',
          type: 'Resolve',
          played_by: 0,
          source: 'Hand',
          requires_additional_input: false,
          card: null,
          target: null,
        },
      ],
    }

    await route.fulfill({ json: payload })
  })

  await page.goto('/')

  await expect(page.locator('.modal-title')).toHaveText('Resolve One-Off')
  await expect(
    page.getByRole('button', { name: 'Resolve one-off Five of Hearts' }),
  ).toBeVisible()

  const responsePromise = page.waitForResponse(
    '**/api/sessions/modal-session/actions',
  )
  await page.locator('.modal').getByRole('button', { name: 'Confirm' }).click()
  await responsePromise
})

test('discard selection flow for three', async ({ page }) => {
  await page.route('**/api/sessions', async (route) => {
    const payload = {
      session_id: 'discard-session',
      state: {
        hands: [[], []],
        hand_counts: [0, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 14,
        discard_pile: [
          {
            id: 'discard-1',
            suit: 'DIAMONDS',
            rank: 'TEN',
            display: 'Ten of Diamonds',
            played_by: null,
            purpose: null,
            point_value: 10,
            is_stolen: false,
            attachments: [],
          },
        ],
        discard_count: 1,
        scores: [0, 0],
        targets: [21, 21],
        turn: 0,
        current_action_player: 0,
        status: null,
        resolving_two: false,
        resolving_one_off: false,
        resolving_three: true,
        overall_turn: 2,
        use_ai: true,
        one_off_card_to_counter: null,
      },
      legal_actions: [
        {
          id: 0,
          label: 'Take Ten of Diamonds from discard',
          type: 'Take From Discard',
          played_by: 0,
          source: 'Discard',
          requires_additional_input: false,
          card: {
            id: 'discard-1',
            suit: 'DIAMONDS',
            rank: 'TEN',
            display: 'Ten of Diamonds',
            played_by: null,
            purpose: null,
            point_value: 10,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
      ],
      state_version: 0,
      ai_thinking: false,
    }

    await route.fulfill({ json: payload })
  })

  await page.route('**/api/sessions/discard-session/history', async (route) => {
    await route.fulfill({ json: { entries: [], turn_counter: 2 } })
  })

  await page.route('**/api/sessions/discard-session', async (route) => {
    const payload = {
      session_id: 'discard-session',
      state: {
        hands: [[], []],
        hand_counts: [0, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 14,
        discard_pile: [
          {
            id: 'discard-1',
            suit: 'DIAMONDS',
            rank: 'TEN',
            display: 'Ten of Diamonds',
            played_by: null,
            purpose: null,
            point_value: 10,
            is_stolen: false,
            attachments: [],
          },
        ],
        discard_count: 1,
        scores: [0, 0],
        targets: [21, 21],
        turn: 0,
        current_action_player: 0,
        status: null,
        resolving_two: false,
        resolving_one_off: false,
        resolving_three: true,
        overall_turn: 2,
        use_ai: true,
        one_off_card_to_counter: null,
      },
      legal_actions: [
        {
          id: 0,
          label: 'Take Ten of Diamonds from discard',
          type: 'Take From Discard',
          played_by: 0,
          source: 'Discard',
          requires_additional_input: false,
          card: {
            id: 'discard-1',
            suit: 'DIAMONDS',
            rank: 'TEN',
            display: 'Ten of Diamonds',
            played_by: null,
            purpose: null,
            point_value: 10,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
      ],
      state_version: 0,
      ai_thinking: false,
    }

    await route.fulfill({ json: payload })
  })

  await page.route('**/api/sessions/discard-session/actions', async (route) => {
    const payload = {
      state: {
        hands: [
          [
            {
              id: 'discard-1',
              suit: 'DIAMONDS',
              rank: 'TEN',
              display: 'Ten of Diamonds',
              played_by: null,
              purpose: null,
              point_value: 10,
              is_stolen: false,
              attachments: [],
            },
          ],
          [],
        ],
        hand_counts: [1, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 14,
        discard_pile: [],
        discard_count: 0,
        scores: [0, 0],
        targets: [21, 21],
        turn: 1,
        current_action_player: 1,
        status: null,
        resolving_two: false,
        resolving_one_off: false,
        resolving_three: false,
        overall_turn: 2,
        use_ai: true,
        one_off_card_to_counter: null,
      },
      legal_actions: [],
      state_version: 1,
      last_actions: [
        {
          id: -1,
          label: 'Take Ten of Diamonds from discard',
          type: 'Take From Discard',
          played_by: 0,
          source: 'Discard',
          requires_additional_input: false,
          card: null,
          target: null,
        },
      ],
    }

    await route.fulfill({ json: payload })
  })

  await page.goto('/')

  await expect(page.getByText('Take From Scrap')).toBeVisible()
  await expect(
    page.getByRole('button', { name: 'Ten of Diamonds' }),
  ).toBeVisible()

  const responsePromise = page.waitForResponse(
    '**/api/sessions/discard-session/actions',
  )
  await page.getByRole('button', { name: 'Take card' }).click()
  await responsePromise
})

test('seven reveal modal flow', async ({ page }) => {
  await page.route('**/api/sessions', async (route) => {
    const payload = {
      session_id: 'seven-session',
      state: {
        hands: [[], []],
        hand_counts: [0, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 22,
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
        resolving_seven: true,
        pending_seven_player: 0,
        pending_seven_cards: [
          {
            id: 'seven-top',
            suit: 'HEARTS',
            rank: 'NINE',
            display: 'Nine of Hearts',
            played_by: null,
            purpose: null,
            point_value: 9,
            is_stolen: false,
            attachments: [],
          },
          {
            id: 'seven-second',
            suit: 'CLUBS',
            rank: 'THREE',
            display: 'Three of Clubs',
            played_by: null,
            purpose: null,
            point_value: 3,
            is_stolen: false,
            attachments: [],
          },
        ],
        pending_seven_requires_discard: false,
        resolving_four: false,
        pending_four_count: 0,
        overall_turn: 1,
        use_ai: true,
        one_off_card_to_counter: null,
      },
      legal_actions: [
        {
          id: 0,
          label: 'Play Nine of Hearts as points',
          type: 'Points',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: {
            id: 'seven-top',
            suit: 'HEARTS',
            rank: 'NINE',
            display: 'Nine of Hearts',
            played_by: null,
            purpose: null,
            point_value: 9,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
        {
          id: 1,
          label: 'Play Three of Clubs as one-off',
          type: 'One-Off',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: {
            id: 'seven-second',
            suit: 'CLUBS',
            rank: 'THREE',
            display: 'Three of Clubs',
            played_by: null,
            purpose: null,
            point_value: 3,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
      ],
      state_version: 0,
      ai_thinking: false,
    }

    await route.fulfill({ json: payload })
  })

  await page.route('**/api/sessions/seven-session/history', async (route) => {
    await route.fulfill({ json: { entries: [], turn_counter: 1 } })
  })

  await page.route('**/api/sessions/seven-session', async (route) => {
    const payload = {
      session_id: 'seven-session',
      state: {
        hands: [[], []],
        hand_counts: [0, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 22,
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
        resolving_seven: true,
        pending_seven_player: 0,
        pending_seven_cards: [
          {
            id: 'seven-top',
            suit: 'HEARTS',
            rank: 'NINE',
            display: 'Nine of Hearts',
            played_by: null,
            purpose: null,
            point_value: 9,
            is_stolen: false,
            attachments: [],
          },
          {
            id: 'seven-second',
            suit: 'CLUBS',
            rank: 'THREE',
            display: 'Three of Clubs',
            played_by: null,
            purpose: null,
            point_value: 3,
            is_stolen: false,
            attachments: [],
          },
        ],
        pending_seven_requires_discard: false,
        resolving_four: false,
        pending_four_count: 0,
        overall_turn: 1,
        use_ai: true,
        one_off_card_to_counter: null,
      },
      legal_actions: [
        {
          id: 0,
          label: 'Play Nine of Hearts as points',
          type: 'Points',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: {
            id: 'seven-top',
            suit: 'HEARTS',
            rank: 'NINE',
            display: 'Nine of Hearts',
            played_by: null,
            purpose: null,
            point_value: 9,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
        {
          id: 1,
          label: 'Play Three of Clubs as one-off',
          type: 'One-Off',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: {
            id: 'seven-second',
            suit: 'CLUBS',
            rank: 'THREE',
            display: 'Three of Clubs',
            played_by: null,
            purpose: null,
            point_value: 3,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
      ],
      state_version: 0,
      ai_thinking: false,
    }

    await route.fulfill({ json: payload })
  })

  await page.route('**/api/sessions/seven-session/actions', async (route) => {
    const payload = {
      state: {
        hands: [[], []],
        hand_counts: [0, 0],
        fields: [
          [
            {
              id: 'seven-top',
              suit: 'HEARTS',
              rank: 'NINE',
              display: 'Nine of Hearts',
              played_by: 0,
              purpose: 'POINTS',
              point_value: 9,
              is_stolen: false,
              attachments: [],
            },
          ],
          [],
        ],
        effective_fields: [
          [
            {
              id: 'seven-top',
              suit: 'HEARTS',
              rank: 'NINE',
              display: 'Nine of Hearts',
              played_by: 0,
              purpose: 'POINTS',
              point_value: 9,
              is_stolen: false,
              attachments: [],
            },
          ],
          [],
        ],
        deck_count: 21,
        discard_pile: [],
        discard_count: 0,
        scores: [9, 0],
        targets: [21, 21],
        turn: 1,
        current_action_player: 1,
        status: null,
        resolving_two: false,
        resolving_one_off: false,
        resolving_three: false,
        resolving_seven: false,
        pending_seven_player: null,
        pending_seven_cards: [],
        pending_seven_requires_discard: false,
        resolving_four: false,
        pending_four_count: 0,
        overall_turn: 1,
        use_ai: true,
        one_off_card_to_counter: null,
      },
      legal_actions: [],
      state_version: 1,
      last_actions: [
        {
          id: -1,
          label: 'Play Nine of Hearts as points',
          type: 'Points',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: null,
          target: null,
        },
      ],
    }

    await route.fulfill({ json: payload })
  })

  await page.goto('/')

  await expect(page.locator('.modal-title')).toHaveText('Seven Reveal')
  await expect(
    page.getByRole('button', { name: 'Play Nine of Hearts as points' }),
  ).toBeVisible()

  const responsePromise = page.waitForResponse(
    '**/api/sessions/seven-session/actions',
  )
  await page.getByRole('button', { name: 'Play Nine of Hearts as points' }).click()
  await page.locator('.modal').getByRole('button', { name: 'Confirm' }).click()
  await responsePromise
})

test('seven discard-only modal flow', async ({ page }) => {
  await page.route('**/api/sessions', async (route) => {
    const payload = {
      session_id: 'seven-discard-session',
      state: {
        hands: [[], []],
        hand_counts: [0, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 10,
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
        resolving_seven: true,
        pending_seven_player: 0,
        pending_seven_cards: [
          {
            id: 'seven-jack-1',
            suit: 'HEARTS',
            rank: 'JACK',
            display: 'Jack of Hearts',
            played_by: null,
            purpose: null,
            point_value: 11,
            is_stolen: false,
            attachments: [],
          },
          {
            id: 'seven-jack-2',
            suit: 'SPADES',
            rank: 'JACK',
            display: 'Jack of Spades',
            played_by: null,
            purpose: null,
            point_value: 11,
            is_stolen: false,
            attachments: [],
          },
        ],
        pending_seven_requires_discard: true,
        resolving_four: false,
        pending_four_count: 0,
        overall_turn: 1,
        use_ai: true,
        one_off_card_to_counter: null,
      },
      legal_actions: [
        {
          id: 0,
          label: 'Discard revealed Jack of Hearts',
          type: 'Discard Revealed',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: {
            id: 'seven-jack-1',
            suit: 'HEARTS',
            rank: 'JACK',
            display: 'Jack of Hearts',
            played_by: null,
            purpose: null,
            point_value: 11,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
        {
          id: 1,
          label: 'Discard revealed Jack of Spades',
          type: 'Discard Revealed',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: {
            id: 'seven-jack-2',
            suit: 'SPADES',
            rank: 'JACK',
            display: 'Jack of Spades',
            played_by: null,
            purpose: null,
            point_value: 11,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
      ],
      state_version: 0,
      ai_thinking: false,
    }

    await route.fulfill({ json: payload })
  })

  await page.route('**/api/sessions/seven-discard-session/history', async (route) => {
    await route.fulfill({ json: { entries: [], turn_counter: 1 } })
  })

  await page.route('**/api/sessions/seven-discard-session', async (route) => {
    const payload = {
      session_id: 'seven-discard-session',
      state: {
        hands: [[], []],
        hand_counts: [0, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 10,
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
        resolving_seven: true,
        pending_seven_player: 0,
        pending_seven_cards: [
          {
            id: 'seven-jack-1',
            suit: 'HEARTS',
            rank: 'JACK',
            display: 'Jack of Hearts',
            played_by: null,
            purpose: null,
            point_value: 11,
            is_stolen: false,
            attachments: [],
          },
          {
            id: 'seven-jack-2',
            suit: 'SPADES',
            rank: 'JACK',
            display: 'Jack of Spades',
            played_by: null,
            purpose: null,
            point_value: 11,
            is_stolen: false,
            attachments: [],
          },
        ],
        pending_seven_requires_discard: true,
        resolving_four: false,
        pending_four_count: 0,
        overall_turn: 1,
        use_ai: true,
        one_off_card_to_counter: null,
      },
      legal_actions: [
        {
          id: 0,
          label: 'Discard revealed Jack of Hearts',
          type: 'Discard Revealed',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: {
            id: 'seven-jack-1',
            suit: 'HEARTS',
            rank: 'JACK',
            display: 'Jack of Hearts',
            played_by: null,
            purpose: null,
            point_value: 11,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
        {
          id: 1,
          label: 'Discard revealed Jack of Spades',
          type: 'Discard Revealed',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: {
            id: 'seven-jack-2',
            suit: 'SPADES',
            rank: 'JACK',
            display: 'Jack of Spades',
            played_by: null,
            purpose: null,
            point_value: 11,
            is_stolen: false,
            attachments: [],
          },
          target: null,
        },
      ],
      state_version: 0,
      ai_thinking: false,
    }

    await route.fulfill({ json: payload })
  })

  await page.route('**/api/sessions/seven-discard-session/actions', async (route) => {
    const payload = {
      state: {
        hands: [[], []],
        hand_counts: [0, 0],
        fields: [[], []],
        effective_fields: [[], []],
        deck_count: 9,
        discard_pile: [
          {
            id: 'seven-jack-1',
            suit: 'HEARTS',
            rank: 'JACK',
            display: 'Jack of Hearts',
            played_by: null,
            purpose: null,
            point_value: 11,
            is_stolen: false,
            attachments: [],
          },
        ],
        discard_count: 1,
        scores: [0, 0],
        targets: [21, 21],
        turn: 1,
        current_action_player: 1,
        status: null,
        resolving_two: false,
        resolving_one_off: false,
        resolving_three: false,
        resolving_seven: false,
        pending_seven_player: null,
        pending_seven_cards: [],
        pending_seven_requires_discard: false,
        resolving_four: false,
        pending_four_count: 0,
        overall_turn: 1,
        use_ai: true,
        one_off_card_to_counter: null,
      },
      legal_actions: [],
      state_version: 1,
      last_actions: [
        {
          id: -1,
          label: 'Discard revealed Jack of Hearts',
          type: 'Discard Revealed',
          played_by: 0,
          source: 'Deck',
          requires_additional_input: false,
          card: null,
          target: null,
        },
      ],
    }

    await route.fulfill({ json: payload })
  })

  await page.goto('/')

  await expect(page.locator('.modal-title')).toHaveText('Seven Reveal')
  await expect(
    page.getByRole('button', { name: 'Discard revealed Jack of Hearts' }),
  ).toBeVisible()

  const responsePromise = page.waitForResponse(
    '**/api/sessions/seven-discard-session/actions',
  )
  await page
    .getByRole('button', { name: 'Discard revealed Jack of Hearts' })
    .click()
  await page.locator('.modal').getByRole('button', { name: 'Confirm' }).click()
  await responsePromise
})

test('game over banner restarts session', async ({ page }) => {
  let callCount = 0
  await page.route('**/api/sessions', async (route) => {
    callCount += 1
    const payload =
      callCount === 1
        ? {
            session_id: 'gameover-session',
            state: {
              hands: [[], []],
              hand_counts: [0, 0],
              fields: [[], []],
              effective_fields: [[], []],
              deck_count: 0,
              discard_pile: [],
              discard_count: 0,
              scores: [21, 10],
              targets: [21, 21],
              turn: 0,
              current_action_player: 0,
              status: 'win',
              resolving_two: false,
              resolving_one_off: false,
              resolving_three: false,
              overall_turn: 5,
              use_ai: true,
              one_off_card_to_counter: null,
            },
            legal_actions: [],
            state_version: 0,
            ai_thinking: false,
          }
        : {
            session_id: 'new-session',
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

    await route.fulfill({ json: payload })
  })

  await page.route('**/api/sessions/gameover-session/history', async (route) => {
    await route.fulfill({ json: { entries: [], turn_counter: 5 } })
  })

  await page.route('**/api/sessions/gameover-session', async (route) => {
    await route.fulfill({
      json: {
        session_id: 'gameover-session',
        state: {
          hands: [[], []],
          hand_counts: [0, 0],
          fields: [[], []],
          effective_fields: [[], []],
          deck_count: 0,
          discard_pile: [],
          discard_count: 0,
          scores: [21, 10],
          targets: [21, 21],
          turn: 0,
          current_action_player: 0,
          status: 'win',
          resolving_two: false,
          resolving_one_off: false,
          resolving_three: false,
          overall_turn: 5,
          use_ai: true,
          one_off_card_to_counter: null,
        },
        legal_actions: [],
        state_version: 0,
        ai_thinking: false,
      },
    })
  })

  await page.route('**/api/sessions/new-session/history', async (route) => {
    await route.fulfill({ json: { entries: [], turn_counter: 0 } })
  })

  await page.route('**/api/sessions/new-session', async (route) => {
    await route.fulfill({
      json: {
        session_id: 'new-session',
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
      },
    })
  })

  await page.goto('/')

  await expect(page.locator('.modal-title')).toHaveText('Game Over')
  await page.locator('.modal').getByRole('button', { name: 'New Game' }).click()
  await expect(page.getByText('Game Over')).not.toBeVisible()
})
