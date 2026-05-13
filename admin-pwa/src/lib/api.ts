export type InviteStatus = "active" | "disabled" | "expired" | "fully_used" | "deleted";

export type PayoutCounts = {
  pending: number;
  completed: number;
  failed: number;
  retryable: number;
};

export type InviteCode = {
  publicId: string;
  code: string;
  label: string;
  status: InviteStatus;
  createdBy: string;
  createdAt: string | null;
  updatedAt: string | null;
  expiresAt: string | null;
  maxUses: number;
  currentUses: number;
  assignedRole: string;
  profitSharePercent: number;
  profitShareWallet: string;
  profitShareStartsAt: string | null;
  profitShareEndsAt: string | null;
  profitShareActive: boolean;
  appliesToVaultTypes: string[];
  totalInviteeProfit: number;
  totalPaidToWallet: number;
  payoutCounts: PayoutCounts;
};

export type InviteSummary = {
  totalCodes: number;
  activeCodes: number;
  totalUses: number;
  totalInviteeProfit: number;
  totalPaidToWallet: number;
  failedPayouts: number;
};

export type Payout = {
  publicId: string;
  inviteCodePublicId: string;
  inviteCode: string;
  invitee: string;
  vaultCyclePublicId: string;
  sourceProfitAmount: number;
  profitSharePercent: number;
  payoutAmount: number;
  asset: string;
  destinationWallet: string;
  status: string;
  idempotencyKey: string;
  createdAt: string | null;
  completedAt: string | null;
  failedReason: string;
};

export type Usage = {
  publicId: string;
  invitee: string;
  usedAt: string | null;
  status: string;
  acceptedDisclosureVersion: string;
};

export type AuditLog = {
  publicId: string;
  admin: string;
  action: string;
  entityType: string;
  entityPublicId: string;
  oldValue: Record<string, unknown>;
  newValue: Record<string, unknown>;
  ipAddress: string;
  createdAt: string | null;
  metadata: Record<string, unknown>;
};

export type InviteFormPayload = {
  code?: string;
  codePrefix?: string;
  batchCount?: number;
  label?: string;
  expirationDate?: string;
  maxUses?: number | "";
  assignedRole?: string;
  profitSharePercent: number;
  profitShareWallet: string;
  profitShareStartsAt?: string;
  profitShareEndsAt?: string;
  profitShareActive: boolean;
  appliesToVaultTypes: string[];
  isActive: boolean;
  confirmSensitiveChange?: boolean;
  confirmationReason?: string;
};

type ApiOptions = RequestInit & {
  csrfToken?: string;
};

export class ApiError extends Error {
  code: string;
  status: number;

  constructor(message: string, code: string, status: number) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

const apiBase = () => (process.env.NEXT_PUBLIC_API_BASE_URL || "").replace(/\/$/, "");

export async function apiFetch<T>(path: string, options: ApiOptions = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (options.csrfToken) {
    headers.set("X-CSRF-Token", options.csrfToken);
  }
  const response = await fetch(`${apiBase()}${path}`, {
    ...options,
    headers,
    credentials: "include",
    cache: "no-store"
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data?.ok === false) {
    throw new ApiError(data?.error || "Request failed", data?.code || "request_failed", response.status);
  }
  return data as T;
}

export function formatCurrency(value: number) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value || 0);
}

export function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value || 0);
}

export function formatDate(value: string | null) {
  if (!value) return "None";
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric" }).format(new Date(value));
}
