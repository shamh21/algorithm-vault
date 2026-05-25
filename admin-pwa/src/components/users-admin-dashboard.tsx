"use client";

import { AlertTriangle, ChevronDown, ExternalLink, Loader2, LogOut, RefreshCw, Search, ShieldCheck, UserRound, Users, WalletCards } from "lucide-react";
import { FormEvent, ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import {
  AdminImpersonationLinkResponse,
  AdminSession,
  AdminSignInPayload,
  AdminUser,
  AdminUserWalletBalance,
  AdminUsersResponse,
  AdminUsersSummary,
  ApiError,
  apiFetch,
  formatCurrency,
  formatDate,
  formatNumber
} from "@/lib/api";

type AuthState = "checking" | "unauthenticated" | "authorized" | "access-denied" | "session-error";
type FundedFilter = "all" | "funded" | "empty";
type UserSort = "portfolio_desc" | "username_asc" | "created_desc";

const emptySummary: AdminUsersSummary = {
  totalUsers: 0,
  fundedUsers: 0,
  activeAssetRows: 0,
  portfolioTotalUsd: 0
};

const fundedFilters: Array<{ key: FundedFilter; label: string }> = [
  { key: "all", label: "All" },
  { key: "funded", label: "Funded" },
  { key: "empty", label: "Empty" }
];

const sortOptions: Array<{ value: UserSort; label: string }> = [
  { value: "portfolio_desc", label: "Portfolio" },
  { value: "username_asc", label: "Username" },
  { value: "created_desc", label: "Newest" }
];

export function UsersAdminDashboard() {
  const [authState, setAuthState] = useState<AuthState>("checking");
  const [adminSession, setAdminSession] = useState<AdminSession | null>(null);
  const [csrfToken, setCsrfToken] = useState("");
  const [signInValues, setSignInValues] = useState<AdminSignInPayload>({ username: "", password: "", totpCode: "" });
  const [signingIn, setSigningIn] = useState(false);
  const [signInError, setSignInError] = useState("");
  const [sessionError, setSessionError] = useState("");
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [summary, setSummary] = useState<AdminUsersSummary>(emptySummary);
  const [search, setSearch] = useState("");
  const [funded, setFunded] = useState<FundedFilter>("all");
  const [sort, setSort] = useState<UserSort>("portfolio_desc");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [generatedAt, setGeneratedAt] = useState<string | null>(null);
  const [truncated, setTruncated] = useState(false);
  const [online, setOnline] = useState(() => (typeof navigator === "undefined" ? true : navigator.onLine));
  const [openingUserId, setOpeningUserId] = useState<number | null>(null);

  const queryPath = useMemo(() => {
    const params = new URLSearchParams();
    params.set("funded", funded);
    params.set("sort", sort);
    params.set("limit", "500");
    if (search.trim()) params.set("search", search.trim());
    return `/admin/api/users?${params.toString()}`;
  }, [funded, search, sort]);

  useEffect(() => {
    const handleOnline = () => setOnline(true);
    const handleOffline = () => setOnline(false);
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  const applySession = useCallback((session: AdminSession) => {
    setAdminSession(session);
    setCsrfToken(session.csrfToken || "");
    setSessionError("");
    if (session.authorized) {
      setLoading(true);
      setAuthState("authorized");
      return;
    }
    setUsers([]);
    setSummary(emptySummary);
    setLoading(false);
    setAuthState(session.authenticated ? "access-denied" : "unauthenticated");
  }, []);

  const bootstrap = useCallback(async () => {
    setAuthState("checking");
    setLoading(false);
    setError("");
    setSignInError("");
    setSessionError("");
    try {
      const session = await apiFetch<AdminSession>("/admin/api/session");
      applySession(session);
    } catch (err) {
      setSessionError(errorMessage(err));
      setAuthState("session-error");
    }
  }, [applySession]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void bootstrap();
  }, [bootstrap]);

  const handleAdminApiAuthError = useCallback((err: unknown) => {
    if (err instanceof ApiError && err.status === 401) {
      setAuthState("unauthenticated");
      setCsrfToken("");
      setUsers([]);
      setSummary(emptySummary);
      setSignInError("Session expired. Sign in again.");
      return true;
    }
    if (err instanceof ApiError && err.status === 403) {
      setAuthState("access-denied");
      setUsers([]);
      setSummary(emptySummary);
      return true;
    }
    return false;
  }, []);

  const loadUsers = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await apiFetch<AdminUsersResponse>(queryPath);
      setUsers(response.users);
      setSummary(response.summary || emptySummary);
      setGeneratedAt(response.generatedAt);
      setTruncated(Boolean(response.truncated));
    } catch (err) {
      if (handleAdminApiAuthError(err)) return;
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, [handleAdminApiAuthError, queryPath]);

  useEffect(() => {
    if (authState !== "authorized" || !csrfToken) return;
    const timer = window.setTimeout(() => void loadUsers(), 220);
    return () => window.clearTimeout(timer);
  }, [authState, csrfToken, loadUsers]);

  async function submitAdminSignIn(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!csrfToken || signingIn) return;
    setSigningIn(true);
    setSignInError("");
    setSessionError("");
    try {
      const session = await apiFetch<AdminSession>("/admin/api/sign-in", {
        method: "POST",
        body: JSON.stringify(signInValues),
        csrfToken
      });
      setSignInValues((values) => ({ ...values, password: "", totpCode: "" }));
      applySession(session);
    } catch (err) {
      setSignInValues((values) => ({ ...values, password: "", totpCode: "" }));
      setSignInError(errorMessage(err));
    } finally {
      setSigningIn(false);
    }
  }

  async function signOutAdmin() {
    if (!csrfToken) {
      void bootstrap();
      return;
    }
    setSaving(true);
    setError("");
    setSignInError("");
    try {
      const session = await apiFetch<AdminSession>("/admin/api/sign-out", { method: "POST", csrfToken });
      setSignInValues({ username: "", password: "", totpCode: "" });
      applySession(session);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  async function openAsUser(user: AdminUser) {
    if (!csrfToken || openingUserId) return;
    setOpeningUserId(user.id);
    setError("");
    setNotice("");
    try {
      const response = await apiFetch<AdminImpersonationLinkResponse>(`/admin/api/users/${user.id}/impersonation-link`, {
        method: "POST",
        csrfToken
      });
      const opened = window.open(response.impersonationUrl, "_blank", "noopener,noreferrer");
      if (!opened) {
        window.location.assign(response.impersonationUrl);
        return;
      }
      setNotice(`Opened ${response.target.username} on algvault.app.`);
    } catch (err) {
      if (handleAdminApiAuthError(err)) return;
      setError(errorMessage(err));
    } finally {
      setOpeningUserId(null);
    }
  }

  if (authState === "checking") {
    return (
      <AdminAuthShell>
        <LoadingState label="Checking admin session" />
      </AdminAuthShell>
    );
  }

  if (authState === "session-error") {
    return (
      <AdminAuthShell>
        <StatePanel tone="danger" title="Session unavailable">
          <p>{sessionError || "Admin session could not be checked."}</p>
          <button className="tap-target mt-4 rounded-full bg-white px-4 text-sm font-semibold text-black" type="button" onClick={() => void bootstrap()}>
            Retry
          </button>
        </StatePanel>
      </AdminAuthShell>
    );
  }

  if (authState === "access-denied") {
    return (
      <AdminAuthShell>
        <StatePanel tone="danger" title="Admin access required">
          <p>{adminSession?.admin?.username || "This account"} is signed in but cannot manage users.</p>
          <button className="tap-target mt-4 rounded-full bg-white px-4 text-sm font-semibold text-black" type="button" onClick={() => void signOutAdmin()}>
            {saving ? "Signing out..." : "Sign out"}
          </button>
        </StatePanel>
      </AdminAuthShell>
    );
  }

  if (authState !== "authorized") {
    return (
      <AdminAuthShell>
        <AdminSignIn
          values={signInValues}
          error={signInError || error}
          submitting={signingIn}
          disabled={!csrfToken}
          onChange={setSignInValues}
          onSubmit={submitAdminSignIn}
          onRetrySession={() => void bootstrap()}
        />
      </AdminAuthShell>
    );
  }

  return (
    <main className="ios-safe-page mx-auto min-h-screen w-full max-w-7xl">
      <header className="sticky top-0 z-20 -mx-4 border-b border-white/10 bg-[#070807]/90 px-4 py-3 backdrop-blur md:static md:mx-0 md:border-0 md:bg-transparent md:px-0">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase text-amber-300">Admin</p>
            <h1 className="text-2xl font-semibold text-white md:text-4xl">Users</h1>
          </div>
          <div className="flex items-center gap-2">
            {adminSession?.admin?.username && <span className="hidden text-sm text-white/55 sm:inline">{adminSession.admin.username}</span>}
            <button
              className="tap-target inline-flex items-center justify-center rounded-full border border-white/10 bg-white/5 px-3 text-sm font-semibold text-white"
              onClick={() => void loadUsers()}
              disabled={loading}
              type="button"
              aria-label="Refresh users"
            >
              <RefreshCw className={`h-5 w-5 ${loading ? "animate-spin" : ""}`} />
            </button>
            <button
              className="tap-target inline-flex items-center justify-center rounded-full border border-white/10 bg-white/5 px-3 text-sm font-semibold text-white"
              onClick={() => void signOutAdmin()}
              disabled={saving}
              type="button"
              aria-label="Sign out of admin"
            >
              <LogOut className="h-5 w-5" />
            </button>
          </div>
        </div>
        <AdminNav active="users" />
      </header>

      {!online && (
        <StateBanner tone="warning" icon={<AlertTriangle className="h-5 w-5" />} title="Offline">
          Support links are paused until your connection returns.
        </StateBanner>
      )}
      {error && (
        <StateBanner tone="danger" icon={<AlertTriangle className="h-5 w-5" />} title="Action failed">
          {error}
        </StateBanner>
      )}
      {notice && (
        <StateBanner tone="success" icon={<ShieldCheck className="h-5 w-5" />} title="Support session ready">
          {notice}
        </StateBanner>
      )}

      <section className="mt-5 grid gap-3 md:grid-cols-4">
        <Metric icon={<Users className="h-5 w-5" />} label="Users" value={formatNumber(summary.totalUsers)} />
        <Metric icon={<WalletCards className="h-5 w-5" />} label="Funded" value={formatNumber(summary.fundedUsers)} />
        <Metric icon={<ShieldCheck className="h-5 w-5" />} label="Assets" value={formatNumber(summary.activeAssetRows)} />
        <Metric icon={<WalletCards className="h-5 w-5" />} label="Portfolio" value={formatCurrency(summary.portfolioTotalUsd)} />
      </section>

      <section className="mt-5 rounded-2xl border border-white/10 bg-white/[0.055] p-4">
        <div className="grid gap-3 md:grid-cols-[1fr_auto_auto]">
          <label className="relative">
            <span className="sr-only">Search users</span>
            <Search className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-white/40" />
            <input
              className="h-12 w-full rounded-full border border-white/10 bg-black/25 pl-12 pr-4 text-sm text-white outline-none focus:border-amber-300"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search username"
              autoComplete="off"
            />
          </label>
          <SegmentedControl
            value={funded}
            options={fundedFilters}
            onChange={(value) => setFunded(value as FundedFilter)}
          />
          <label className="relative">
            <span className="sr-only">Sort users</span>
            <select
              className="h-12 appearance-none rounded-full border border-white/10 bg-black/25 pl-4 pr-10 text-sm font-semibold text-white outline-none focus:border-amber-300"
              value={sort}
              onChange={(event) => setSort(event.target.value as UserSort)}
            >
              {sortOptions.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-4 top-1/2 h-5 w-5 -translate-y-1/2 text-white/50" />
          </label>
        </div>
      </section>

      <section className="mt-5">
        {loading ? (
          <LoadingState label="Loading users" />
        ) : users.length === 0 ? (
          <StatePanel tone="neutral" title="No users found">
            <p>Change the search or filter to inspect another account set.</p>
          </StatePanel>
        ) : (
          <>
            {truncated && <p className="mb-3 text-sm text-white/50">Showing the first 500 matching users. Narrow the search for specific accounts.</p>}
            <div className="grid gap-3 lg:hidden">
              {users.map((user) => (
                <UserCard key={user.id} user={user} opening={openingUserId === user.id} onOpen={() => void openAsUser(user)} />
              ))}
            </div>
            <UsersTable users={users} openingUserId={openingUserId} onOpen={(user) => void openAsUser(user)} />
          </>
        )}
        {generatedAt && <p className="mt-4 text-xs text-white/35">Generated {formatDate(generatedAt)}</p>}
      </section>
    </main>
  );
}

function AdminNav({ active }: { active: "invites" | "users" }) {
  const items = [
    { href: "/", label: "Invite Codes", key: "invites" },
    { href: "/users", label: "Users", key: "users" }
  ] as const;
  return (
    <nav className="mt-4 flex gap-2 overflow-x-auto pb-1" aria-label="Admin sections">
      {items.map((item) => (
        <a
          key={item.key}
          href={item.href}
          className={`tap-target inline-flex shrink-0 items-center justify-center rounded-full px-4 text-sm font-semibold ${
            active === item.key ? "bg-white text-black" : "border border-white/10 bg-white/5 text-white"
          }`}
          aria-current={active === item.key ? "page" : undefined}
        >
          {item.label}
        </a>
      ))}
    </nav>
  );
}

function UserCard({ user, opening, onOpen }: { user: AdminUser; opening: boolean; onOpen: () => void }) {
  return (
    <article className="rounded-2xl border border-white/10 bg-white/[0.055] p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <UserRound className="h-5 w-5 shrink-0 text-amber-200" aria-hidden="true" />
            <h2 className="truncate text-lg font-semibold text-white">{user.username}</h2>
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <StatusChip tone={user.role === "admin" ? "warning" : "neutral"}>{user.role}</StatusChip>
            <StatusChip tone={user.twoFactorEnabled ? "success" : "warning"}>{user.twoFactorEnabled ? "2FA" : "No 2FA"}</StatusChip>
          </div>
        </div>
        <div className="text-right">
          <p className="text-lg font-semibold text-white">{formatCurrency(user.wallet.portfolioTotalUsd)}</p>
          <p className="text-xs text-white/45">{formatNumber(user.wallet.activeAssetCount)} assets</p>
        </div>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2">
        <MiniMetric label="Cycles" value={formatNumber(user.activity.activeCyclesCount)} />
        <MiniMetric label="Orders" value={formatNumber(user.activity.activeOrderCount)} />
      </div>
      <button
        className="tap-target mt-4 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-amber-300 px-3 text-sm font-semibold text-black disabled:opacity-60"
        type="button"
        onClick={onOpen}
        disabled={opening}
      >
        {opening ? <Loader2 className="h-4 w-4 animate-spin" /> : <ExternalLink className="h-4 w-4" />}
        Open as user
      </button>
      <BalanceList balances={user.wallet.balances} />
    </article>
  );
}

function UsersTable({ users, openingUserId, onOpen }: { users: AdminUser[]; openingUserId: number | null; onOpen: (user: AdminUser) => void }) {
  return (
    <div className="hidden overflow-hidden rounded-2xl border border-white/10 lg:block">
      <table className="w-full border-collapse text-left text-sm">
        <thead className="bg-white/[0.07] text-xs uppercase text-white/55">
          <tr>
            <th className="px-4 py-3 font-semibold">User</th>
            <th className="px-4 py-3 font-semibold">Wallet</th>
            <th className="px-4 py-3 font-semibold">Activity</th>
            <th className="px-4 py-3 font-semibold">Balances</th>
            <th className="px-4 py-3 font-semibold">Support</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/10">
          {users.map((user) => (
            <tr key={user.id} className="align-top">
              <td className="px-4 py-4">
                <div className="font-semibold text-white">{user.username}</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  <StatusChip tone={user.role === "admin" ? "warning" : "neutral"}>{user.role}</StatusChip>
                  <StatusChip tone={user.twoFactorEnabled ? "success" : "warning"}>{user.twoFactorEnabled ? "2FA enabled" : "No 2FA"}</StatusChip>
                </div>
                <p className="mt-2 text-xs text-white/40">Joined {formatDate(user.createdAt)}</p>
              </td>
              <td className="px-4 py-4">
                <div className="font-semibold text-white">{formatCurrency(user.wallet.portfolioTotalUsd)}</div>
                <p className="mt-1 text-xs text-white/45">{formatNumber(user.wallet.activeAssetCount)} active assets</p>
              </td>
              <td className="px-4 py-4 text-white/70">
                <p>{formatNumber(user.activity.activeCyclesCount)} active cycles</p>
                <p>{formatNumber(user.activity.activeOrderCount)} active orders</p>
              </td>
              <td className="px-4 py-4">
                <details>
                  <summary className="cursor-pointer text-sm font-semibold text-amber-200">View assets</summary>
                  <BalanceList balances={user.wallet.balances} compact />
                </details>
              </td>
              <td className="px-4 py-4">
                <button
                  className="tap-target inline-flex items-center justify-center gap-2 rounded-full bg-amber-300 px-4 text-sm font-semibold text-black disabled:opacity-60"
                  type="button"
                  onClick={() => onOpen(user)}
                  disabled={openingUserId === user.id}
                >
                  {openingUserId === user.id ? <Loader2 className="h-4 w-4 animate-spin" /> : <ExternalLink className="h-4 w-4" />}
                  Open as
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BalanceList({ balances, compact = false }: { balances: AdminUserWalletBalance[]; compact?: boolean }) {
  if (balances.length === 0) {
    return <p className="mt-3 rounded-xl border border-white/10 bg-black/20 p-3 text-sm text-white/50">No active wallet balances.</p>;
  }
  return (
    <div className={`mt-3 grid gap-2 ${compact ? "" : "sm:grid-cols-2"}`}>
      {balances.map((balance) => (
        <div key={balance.asset} className="rounded-xl border border-white/10 bg-black/20 p-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="font-semibold text-white">{balance.asset}</p>
              <p className="mt-1 text-xs text-white/45">{formatCurrency(balance.estimatedUsdValue)}</p>
            </div>
            <StatusChip tone={statusTone(balance.syncStatus, balance.onchainStatus)}>{statusLabel(balance.syncStatus, balance.onchainStatus)}</StatusChip>
          </div>
          <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
            <MiniMetric label="Available" value={formatAssetAmount(balance.availableBalance)} />
            <MiniMetric label="Locked" value={formatAssetAmount(balance.lockedBalance)} />
            <MiniMetric label="Total" value={formatAssetAmount(balance.totalBalance)} />
          </div>
          {balance.onchainCheckedAt && <p className="mt-2 text-xs text-white/40">Checked {formatDate(balance.onchainCheckedAt)}</p>}
        </div>
      ))}
    </div>
  );
}

function Metric({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.055] p-4">
      <div className="flex items-center gap-2 text-amber-200">
        {icon}
        <span className="text-xs font-semibold uppercase">{label}</span>
      </div>
      <p className="mt-3 truncate text-2xl font-semibold text-white">{value}</p>
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/20 p-2">
      <p className="text-[11px] uppercase text-white/40">{label}</p>
      <p className="mt-1 truncate text-sm font-semibold text-white">{value}</p>
    </div>
  );
}

function SegmentedControl({ value, options, onChange }: { value: string; options: Array<{ key: string; label: string }>; onChange: (value: string) => void }) {
  return (
    <div className="inline-flex h-12 rounded-full border border-white/10 bg-black/25 p-1">
      {options.map((option) => (
        <button
          key={option.key}
          type="button"
          className={`rounded-full px-4 text-sm font-semibold ${value === option.key ? "bg-white text-black" : "text-white/65"}`}
          onClick={() => onChange(option.key)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function AdminAuthShell({ children }: { children: ReactNode }) {
  return (
    <main className="ios-safe-page flex min-h-screen items-center justify-center">
      <div className="w-full max-w-md">{children}</div>
    </main>
  );
}

function AdminSignIn({
  values,
  error,
  submitting,
  disabled,
  onChange,
  onSubmit,
  onRetrySession
}: {
  values: AdminSignInPayload;
  error: string;
  submitting: boolean;
  disabled: boolean;
  onChange: (values: AdminSignInPayload) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onRetrySession: () => void;
}) {
  const totpCode = values.totpCode.trim();
  const canSubmit = Boolean(values.username.trim() && values.password && /^\d{6}$/.test(totpCode) && !submitting && !disabled);
  return (
    <section className="rounded-3xl border border-white/10 bg-white/[0.06] p-6 shadow-2xl shadow-black/30">
      <p className="text-xs font-semibold uppercase text-amber-300">Admin</p>
      <h1 className="mt-2 text-3xl font-semibold text-white">Users access</h1>
      <p className="mt-2 text-sm text-white/55">Sign in with an admin account and authenticator code.</p>
      {error && (
        <div className="mt-4 rounded-2xl border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-100">
          {error}
          <button className="ml-2 font-semibold underline" type="button" onClick={onRetrySession}>
            Retry
          </button>
        </div>
      )}
      <form className="mt-5 grid gap-4" onSubmit={onSubmit}>
        <Field label="Username">
          <input
            className="h-12 w-full rounded-2xl border border-white/10 bg-black/25 px-4 text-white outline-none focus:border-amber-300"
            value={values.username}
            autoComplete="username"
            onChange={(event) => onChange({ ...values, username: event.target.value })}
          />
        </Field>
        <Field label="Password">
          <input
            className="h-12 w-full rounded-2xl border border-white/10 bg-black/25 px-4 text-white outline-none focus:border-amber-300"
            type="password"
            value={values.password}
            autoComplete="current-password"
            onChange={(event) => onChange({ ...values, password: event.target.value })}
          />
        </Field>
        <Field label="TOTP code">
          <input
            className="h-12 w-full rounded-2xl border border-white/10 bg-black/25 px-4 text-white outline-none focus:border-amber-300"
            value={values.totpCode}
            inputMode="numeric"
            pattern="[0-9]{6}"
            maxLength={6}
            autoComplete="one-time-code"
            onChange={(event) => onChange({ ...values, totpCode: event.target.value.replace(/\D/g, "").slice(0, 6) })}
          />
        </Field>
        <button
          className="tap-target inline-flex items-center justify-center gap-2 rounded-full bg-amber-300 px-4 font-semibold text-black disabled:cursor-not-allowed disabled:opacity-50"
          type="submit"
          disabled={!canSubmit}
        >
          {submitting && <Loader2 className="h-5 w-5 animate-spin" />}
          Sign in
        </button>
      </form>
    </section>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="grid gap-2">
      <span className="text-sm font-semibold text-white/75">{label}</span>
      {children}
    </label>
  );
}

function LoadingState({ label }: { label: string }) {
  return (
    <div className="flex min-h-[240px] items-center justify-center rounded-2xl border border-white/10 bg-white/[0.055]">
      <div className="flex items-center gap-3 text-white/70">
        <Loader2 className="h-6 w-6 animate-spin text-amber-300" />
        <span>{label}</span>
      </div>
    </div>
  );
}

function StatePanel({ tone, title, children }: { tone: "danger" | "neutral"; title: string; children: ReactNode }) {
  const classes = tone === "danger" ? "border-red-400/30 bg-red-500/10 text-red-100" : "border-white/10 bg-white/[0.055] text-white/60";
  return (
    <section className={`rounded-2xl border p-5 ${classes}`}>
      <h2 className="text-lg font-semibold text-white">{title}</h2>
      <div className="mt-2 text-sm">{children}</div>
    </section>
  );
}

function StateBanner({ tone, icon, title, children }: { tone: "success" | "warning" | "danger"; icon: ReactNode; title: string; children: ReactNode }) {
  const classes =
    tone === "success"
      ? "border-emerald-400/30 bg-emerald-500/10 text-emerald-100"
      : tone === "warning"
        ? "border-amber-300/30 bg-amber-400/10 text-amber-100"
        : "border-red-400/30 bg-red-500/10 text-red-100";
  return (
    <section className={`mt-4 flex items-start gap-3 rounded-2xl border p-4 ${classes}`}>
      <div className="mt-0.5">{icon}</div>
      <div>
        <h2 className="font-semibold text-white">{title}</h2>
        <p className="mt-1 text-sm">{children}</p>
      </div>
    </section>
  );
}

function StatusChip({ tone, children }: { tone: "success" | "warning" | "neutral"; children: ReactNode }) {
  const classes =
    tone === "success"
      ? "border-emerald-400/30 bg-emerald-500/10 text-emerald-100"
      : tone === "warning"
        ? "border-amber-300/30 bg-amber-400/10 text-amber-100"
        : "border-white/10 bg-white/5 text-white/65";
  return <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold ${classes}`}>{children}</span>;
}

function statusTone(syncStatus: string, onchainStatus: string): "success" | "warning" | "neutral" {
  if (syncStatus === "synced" || onchainStatus === "checked") return "success";
  if (syncStatus === "failed" || onchainStatus === "failed") return "warning";
  return "neutral";
}

function statusLabel(syncStatus: string, onchainStatus: string) {
  if (onchainStatus === "checked") return "Checked";
  if (syncStatus === "synced") return "Synced";
  if (syncStatus === "failed" || onchainStatus === "failed") return "Attention";
  return "Pending";
}

function formatAssetAmount(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 6 }).format(value || 0);
}

function errorMessage(err: unknown) {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Request failed";
}
