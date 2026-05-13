# Invite-Code Profit Share Admin PWA

## Product Requirements

Build a mobile-first admin Progressive Web App for iPhone Safari and installed iOS PWA use. Admins create, batch-generate, search, filter, sort, disable, delete, inspect, and audit invite codes. Each invite code may define an invitee profit-share rule that allocates a configured percentage of positive completed Vault Cycle profit to the destination wallet `sufyanh`.

Locked business rules:

- Profit share is calculated only from positive completed Vault Cycle profit.
- Principal, deposits, and losses are never charged.
- Destination wallet defaults to `sufyanh`.
- Disabled, expired, fully used, and deleted codes cannot be used for new signups.
- Existing invitees continue under the current invite-code rule unless the profit-share rule is inactive or out of date range/scope.
- Admin changes to percent or wallet require explicit confirmation and apply only to future completed cycles.
- Payout and audit records are append-only for reconciliation.

Example:

- Invitee joins with `SUFYAN20`.
- Invite code has a 10% profit-share rule.
- Invitee earns `$500` positive profit in a completed Vault Cycle.
- `$50` is credited to `sufyanh`.
- `$450` remains with the invitee before other applicable fees.

## Admin User Flow

1. Admin opens the PWA and existing admin auth/session is checked through `/admin/api/session`.
2. Dashboard loads summary metrics, invite code list, current filters, and status counts.
3. Admin taps `Create` or `Batch`.
4. Bottom sheet opens with code, label, expiration, max uses, role, percent, wallet, start/end dates, vault type scope, and active controls.
5. Backend validates all rule fields server-side and creates one or more invite codes.
6. Admin can tap a code to open details tabs: Overview, Invitees, Payouts, Audit.
7. Admin can copy, edit, disable, delete, export, search, filter, and sort.
8. Sensitive edits to profit-share percent or wallet show confirmation copy before PATCH.
9. During settlement, completed cycles run idempotent invite profit-share processing before final invitee credit.

## Mobile Screen Layout

The PWA is implemented as a single App Router dashboard screen:

- Sticky top header with title and refresh action.
- Offline/error notice band for unsafe states.
- Five metric tiles: active codes, uses, invitee profit, paid to `sufyanh`, failed payouts.
- Horizontal segmented status filter optimized for thumb reach.
- Search input and sort menu.
- Mobile card list with code, label, status, role, wallet, profit share, usage, profit, payout, copy/edit/actions.
- Desktop table for wider layouts.
- Sticky iOS-safe bottom action bar with `Create`, `Batch`, and `Export`.
- Bottom sheet for create/edit/batch forms.
- Details sheet with Overview, Invitees, Payouts, and Audit tabs.

iOS PWA design rules:

- Uses `viewport-fit=cover`.
- Applies top/bottom safe-area padding.
- Touch targets are at least 44px high.
- No hover-only controls.
- Form actions are reachable from the bottom of the screen.
- Empty, loading, offline, warning, and failure states are visible.

## Component Breakdown

Frontend implementation:

- `admin-pwa/src/app/layout.tsx`: PWA metadata, iOS viewport, theme color, manifest links.
- `admin-pwa/src/app/manifest.ts`: installable PWA manifest.
- `admin-pwa/src/app/page.tsx`: dashboard route.
- `admin-pwa/src/app/globals.css`: Tailwind v4 theme, safe-area helpers, touch target sizing.
- `admin-pwa/src/lib/api.ts`: typed API client, serializers, date/currency helpers, API error handling.
- `admin-pwa/src/components/invite-admin-dashboard.tsx`: dashboard state, list/table, sheets, tabs, forms, copy/export, confirmations.

Backend implementation:

- `app/models.py`: invite-code, usage, payout, Vault Cycle public ID, and admin audit models.
- `app/routes/admin.py`: JSON admin APIs under `/admin/api/...` and payout trigger route.
- `app/routes/auth.py`: invite-code signup validation, role assignment, disclosure acceptance.
- `app/services/invite_profit_share.py`: idempotent profit-share calculation, wallet credit, invitee history, payout ledger, audit writes.
- `app/services/vault_cycle_settlement.py`: settlement integration before final invitee wallet credit.
- `app/services/platform_treasury.py`: legacy default referral profit share disabled unless invite-linked.
- `migrations/versions/c7f3a2b9d8e1_invite_profit_share_admin_pwa.py`: schema migration.
- `tests/test_invite_profit_share.py`: positive, zero, loss, duplicate, and failure payout coverage.

## Data Model

### InviteCode

Implemented by extending `ReferralInviteCode`:

- `id`: internal numeric primary key, never exposed through admin JSON.
- `public_id`: external stable ID.
- `code`: normalized uppercase invite code.
- `label`: optional campaign/display label.
- `created_by_user_id`, `created_at`.
- `expires_at`.
- `max_uses`, `usage_count`.
- `is_active`, `disabled_at`, `deleted_at`.
- `assigned_role`.
- `profit_share_percent`.
- `profit_share_wallet`, default `sufyanh`.
- `profit_share_starts_at`, `profit_share_ends_at`.
- `profit_share_active`.
- `applies_to_vault_types_json`.

Computed status:

- `active`
- `disabled`
- `expired`
- `fully_used`
- `deleted`

### InviteCodeUsage

Records the one-time signup relationship:

- `public_id`
- `invite_code_id`
- `invitee_user_id`
- `used_at`
- `status`
- `accepted_disclosure_version`
- `metadata_json`

### VaultCycle

Existing model extended with:

- `public_id`
- `user_id`
- `vault_id`
- cycle start/end timestamps
- principal/ending/final settlement fields
- net/gross PnL fields
- `status`

### ProfitSharePayout

Append-only payout ledger:

- `public_id`
- `invite_code_id`
- `invitee_user_id`
- `destination_user_id`
- `vault_cycle_id`
- `vault_cycle_settlement_id`
- `asset`
- `source_profit_amount`
- `profit_share_percent`
- `payout_amount`
- `destination_wallet`
- `status`: `pending`, `completed`, `failed`, `retryable`, `reversed`
- `idempotency_key`
- `created_at`, `completed_at`
- `failed_reason`
- `details_json`

Unique idempotency key prevents duplicate payouts for the same cycle/code pair.

### AdminAuditLog

Admin-facing immutable audit trail:

- `public_id`
- `admin_user_id`
- `admin_username`
- `action`
- `entity_type`
- `entity_public_id`
- `old_value_json`
- `new_value_json`
- `metadata_json`
- `ip_address`
- `user_agent`
- `created_at`

## API Routes

Actual implementation uses `/admin/api` so existing Jinja admin routes are not broken.

- `GET /admin/api/session`
- `POST /admin/api/invite-codes`
- `GET /admin/api/invite-codes`
- `GET /admin/api/invite-codes/:publicId`
- `PATCH /admin/api/invite-codes/:publicId`
- `POST /admin/api/invite-codes/:publicId/disable`
- `DELETE /admin/api/invite-codes/:publicId`
- `GET /admin/api/invite-codes/:publicId/usages`
- `GET /admin/api/invite-codes/:publicId/profit-share-payouts`
- `POST /api/vault-cycles/:publicId/process-profit-share`
- `GET /admin/api/profit-share-payouts`
- `GET /admin/api/audit-logs`

JSON responses expose public IDs, code values, display wallet names, lifecycle status, and aggregate metrics. Internal numeric IDs are not exposed.

## Backend Payout Logic

Settlement integration:

1. Vault Cycle settlement must be complete.
2. Compute `source_profit_amount = max(settlement/cycle net profit, 0)`.
3. If source profit is zero or negative, skip payout and audit the skipped reason.
4. Load the invitee's accepted invite code usage.
5. Load current invite-code profit-share rule.
6. Verify rule is active, in date range, and applies to the cycle vault type.
7. Calculate `payout_amount = source_profit_amount * profit_share_percent / 100`.
8. Cap payout to the post-treasury positive credit amount so principal is never touched.
9. Use idempotency key:
   `invite-profit-share:vault-cycle:{cycle_public_id}:invite-code:{invite_public_id}`.
10. In one DB transaction:
    - create `ProfitSharePayout`
    - credit `sufyanh` wallet balance
    - append destination wallet transaction
    - append invitee-visible deduction/history transaction
    - reduce invitee final settlement credit by payout amount
    - write audit log
11. Duplicate processing returns the existing payout.
12. Failed/retryable prior payout attempts block silent deduction and return a recovery error.

Failure handling:

- Missing destination wallet creates failed/retryable payout metadata.
- Wallet credit errors keep settlement in recovery.
- No deduction is silently applied if destination credit fails.
- Completed payout rows are never mutated for corrections; reversal/correction entries must be appended.

## Validation Rules

Server-side validation enforces:

- Invite code must be unique after uppercase normalization.
- Profit share percent must be between `0` and `100`.
- Wallet is required when percent is greater than `0`.
- Wallet defaults to `sufyanh`.
- Expiration date cannot be in the past.
- Max uses must be a positive integer when supplied.
- Disabled invite codes cannot be used.
- Deleted invite codes cannot be used.
- Expired invite codes cannot be used for new signups.
- Fully used invite codes cannot be used for new signups.
- A user cannot apply multiple invite codes.
- Profit share cannot be applied twice to the same Vault Cycle.
- Profit share applies only to positive profit.
- Profit share is capped so principal is never reduced.
- Sensitive admin updates require confirmation.

## Edge Cases

- No profit: no payout, skipped calculation recorded.
- Loss: no payout, skipped calculation recorded.
- Code expires after signup: blocks new signups only.
- Code disabled after signup: blocks new signups; existing invitees continue if the profit-share rule remains active.
- Profit-share rule inactive: no future payout.
- Admin changes percent/wallet: confirmation required; applies to future completed cycles.
- Wallet payout fails: payout is marked failed/retryable and settlement remains recoverable.
- Duplicate processing request: returns existing completed payout or blocks retryable/failed rows.
- Multiple invite codes attempted by same user: signup rejects additional code usage.
- Invitee changes vaults: payout follows invitee usage and vault-type scope.
- Partial cycle completion: no payout.
- Refunds/corrections/recalculation: append reversal/correction ledger entries; do not mutate completed payout rows.

## UI Copy

Signup disclosure:

> Using this invite code may allocate a percentage of positive Vault Cycle profit to `sufyanh`. It never applies to your deposits, principal, or losses.

Admin sensitive-change confirmation:

> Changing this profit-share rule applies to future completed Vault Cycles for all users who joined with this code.

Invitee transaction label:

> Invite code profit share: {percent}% of positive Vault Cycle profit paid to `sufyanh`.

Empty state:

> No invite codes yet. Create one code or batch-generate a set for a campaign.

Failed payout state:

> Wallet credit failed. The payout was not completed and needs review.

## Security Notes

- Uses existing admin auth and admin route guards.
- Requires CSRF token for mutating admin API requests.
- Validates profit-share rules server-side.
- Payout math uses `Decimal` with fixed precision.
- Never trusts client-submitted payout percentages during settlement.
- Uses public IDs in admin APIs.
- Logs admin action, old/new values, IP address, and user agent.
- Logs payout calculation details and skipped reasons.
- Uses idempotency keys for cycle payouts.
- Keeps payout/audit rows append-only for reconciliation.
- Requires explicit confirmation for percent or destination wallet changes.
- Keeps signup disclosure acceptance metadata on invite-code usage.

## Implementation And Operations

Local PWA:

```bash
cd /Users/hishamhassan/Documents/TradingBot/admin-pwa
BACKEND_ORIGIN=http://localhost:5000 npx -y vercel@50.28.0 dev --listen 3010 --yes
```

Standalone Next.js checks:

```bash
cd /Users/hishamhassan/Documents/TradingBot/admin-pwa
npm run lint
npm run build
npm audit --omit=dev
```

Focused backend checks:

```bash
cd /Users/hishamhassan/Documents/TradingBot
.venv/bin/python -m compileall app tests/test_invite_profit_share.py
.venv/bin/python -m ruff check app/models.py app/routes/admin.py app/routes/auth.py app/services/invite_profit_share.py app/services/platform_treasury.py app/services/vault_cycle_settlement.py tests/test_invite_profit_share.py tests/test_vault_cycle_engine.py migrations/versions/c7f3a2b9d8e1_invite_profit_share_admin_pwa.py
.venv/bin/python -m pytest tests/test_invite_profit_share.py tests/test_vault_cycle_engine.py::test_vault_settlement_deducts_gas_reserve_without_legacy_referral_profit_share tests/test_vault_cycle_engine.py::test_vault_settlement_without_referral_has_no_default_profit_share tests/test_vault_cycle_engine.py::test_vault_settlement_skips_profit_share_when_gas_deduction_removes_profit
```

Vercel:

- PWA project: `admin-pwa`.
- Root Flask project remains `algorithm-vault`.
- `admin-pwa/` is excluded from root Flask deploys through `.vercelignore`.
- PWA requires `BACKEND_ORIGIN` to point at the deployed Flask backend that contains the new `/admin/api/...` routes.
