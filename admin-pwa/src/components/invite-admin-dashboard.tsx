"use client";

import {
  AlertTriangle,
  CalendarClock,
  CheckCircle2,
  ChevronDown,
  Clipboard,
  Copy,
  Download,
  Edit3,
  Loader2,
  MoreHorizontal,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  Trash2,
  Wallet,
  X
} from "lucide-react";
import { FormEvent, ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import {
  ApiError,
  AuditLog,
  InviteCode,
  InviteFormPayload,
  InviteStatus,
  InviteSummary,
  Payout,
  Usage,
  apiFetch,
  formatCurrency,
  formatDate,
  formatNumber
} from "@/lib/api";

const statuses: Array<{ key: "all" | InviteStatus; label: string }> = [
  { key: "all", label: "All" },
  { key: "active", label: "Active" },
  { key: "disabled", label: "Disabled" },
  { key: "expired", label: "Expired" },
  { key: "fully_used", label: "Full" },
  { key: "deleted", label: "Deleted" }
];

const sortOptions = [
  ["created_desc", "Newest"],
  ["created_asc", "Oldest"],
  ["expiration_asc", "Expiration"],
  ["usage_desc", "Most used"],
  ["profit_desc", "Profit"],
  ["payout_desc", "Payout"]
];

const vaultTypeOptions = ["VaultCycle", "1H10", "Balanced", "Aggressive"];

const emptySummary: InviteSummary = {
  totalCodes: 0,
  activeCodes: 0,
  totalUses: 0,
  totalInviteeProfit: 0,
  totalPaidToWallet: 0,
  failedPayouts: 0
};

const defaultForm = (): InviteFormPayload => ({
  code: "",
  codePrefix: "",
  batchCount: 1,
  label: "",
  expirationDate: "",
  maxUses: "",
  assignedRole: "user",
  profitSharePercent: 10,
  profitShareWallet: "sufyanh",
  profitShareStartsAt: "",
  profitShareEndsAt: "",
  profitShareActive: true,
  appliesToVaultTypes: [],
  isActive: true
});

export function InviteAdminDashboard() {
  const [csrfToken, setCsrfToken] = useState("");
  const [codes, setCodes] = useState<InviteCode[]>([]);
  const [summary, setSummary] = useState<InviteSummary>(emptySummary);
  const [status, setStatus] = useState<"all" | InviteStatus>("all");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("created_desc");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [online, setOnline] = useState(() => (typeof navigator === "undefined" ? true : navigator.onLine));
  const [sheet, setSheet] = useState<"closed" | "create" | "batch" | "edit">("closed");
  const [form, setForm] = useState<InviteFormPayload>(defaultForm);
  const [editing, setEditing] = useState<InviteCode | null>(null);
  const [details, setDetails] = useState<InviteCode | null>(null);
  const [detailTab, setDetailTab] = useState<"overview" | "invitees" | "payouts" | "audit">("overview");
  const [detailData, setDetailData] = useState<{ usages: Usage[]; payouts: Payout[]; auditLogs: AuditLog[] }>({
    usages: [],
    payouts: [],
    auditLogs: []
  });
  const [copiedCode, setCopiedCode] = useState("");

  const queryPath = useMemo(() => {
    const params = new URLSearchParams();
    params.set("status", status);
    params.set("sort", sort);
    if (search.trim()) params.set("search", search.trim());
    return `/admin/api/invite-codes?${params.toString()}`;
  }, [search, sort, status]);

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

  async function bootstrap() {
    setLoading(true);
    setError("");
    try {
      const session = await apiFetch<{ ok: true; csrfToken: string }>("/admin/api/session");
      setCsrfToken(session.csrfToken);
    } catch (err) {
      setError(errorMessage(err));
      setLoading(false);
    }
  }

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void bootstrap();
  }, []);

  const loadInviteCodes = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await apiFetch<{ ok: true; inviteCodes: InviteCode[]; summary: InviteSummary }>(queryPath);
      setCodes(response.inviteCodes);
      setSummary(response.summary || emptySummary);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, [queryPath]);

  useEffect(() => {
    if (!csrfToken) return;
    const timer = window.setTimeout(() => void loadInviteCodes(), 220);
    return () => window.clearTimeout(timer);
  }, [csrfToken, loadInviteCodes]);

  async function openDetails(invite: InviteCode) {
    setDetails(invite);
    setDetailTab("overview");
    setDetailData({ usages: [], payouts: [], auditLogs: [] });
    try {
      const [usages, payouts, audits] = await Promise.all([
        apiFetch<{ ok: true; usages: Usage[] }>(`/admin/api/invite-codes/${invite.publicId}/usages`),
        apiFetch<{ ok: true; payouts: Payout[] }>(`/admin/api/invite-codes/${invite.publicId}/profit-share-payouts`),
        apiFetch<{ ok: true; auditLogs: AuditLog[] }>(`/admin/api/audit-logs?entityPublicId=${invite.publicId}`)
      ]);
      setDetailData({ usages: usages.usages, payouts: payouts.payouts, auditLogs: audits.auditLogs });
    } catch (err) {
      setNotice(errorMessage(err));
    }
  }

  function startCreate(mode: "create" | "batch") {
    setEditing(null);
    setForm({ ...defaultForm(), batchCount: mode === "batch" ? 5 : 1 });
    setSheet(mode);
  }

  function startEdit(invite: InviteCode) {
    setEditing(invite);
    setForm({
      code: invite.code,
      label: invite.label,
      expirationDate: toDateInput(invite.expiresAt),
      maxUses: invite.maxUses || "",
      assignedRole: invite.assignedRole || "user",
      profitSharePercent: invite.profitSharePercent,
      profitShareWallet: invite.profitShareWallet || "sufyanh",
      profitShareStartsAt: toDateTimeInput(invite.profitShareStartsAt),
      profitShareEndsAt: toDateTimeInput(invite.profitShareEndsAt),
      profitShareActive: invite.profitShareActive,
      appliesToVaultTypes: invite.appliesToVaultTypes || [],
      isActive: invite.status !== "disabled" && invite.status !== "deleted"
    });
    setSheet("edit");
  }

  async function submitForm(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (sheet === "closed") return;
    setSaving(true);
    setNotice("");
    setError("");
    const payload = normalizeFormPayload(form, sheet);
    try {
      if (sheet === "edit" && editing) {
        const sensitiveChanged =
          Number(payload.profitSharePercent) !== Number(editing.profitSharePercent) ||
          String(payload.profitShareWallet).toLowerCase() !== String(editing.profitShareWallet).toLowerCase();
        if (sensitiveChanged) {
          const confirmed = window.confirm(
            "Changing this profit-share rule applies to future completed Vault Cycles for all users who joined with this code."
          );
          if (!confirmed) return;
          payload.confirmSensitiveChange = true;
          payload.confirmationReason = "Admin confirmed future-cycle profit-share rule change.";
        }
        await apiFetch(`/admin/api/invite-codes/${editing.publicId}`, {
          method: "PATCH",
          body: JSON.stringify(payload),
          csrfToken
        });
        setNotice("Invite code updated.");
      } else {
        await apiFetch("/admin/api/invite-codes", {
          method: "POST",
          body: JSON.stringify(payload),
          csrfToken
        });
        setNotice(sheet === "batch" ? "Invite codes generated." : "Invite code created.");
      }
      setSheet("closed");
      await loadInviteCodes();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  async function disableInvite(invite: InviteCode) {
    const confirmed = window.confirm(`Disable invite code ${invite.code}? Existing invitees keep future-cycle profit-share rules unless the rule is made inactive.`);
    if (!confirmed) return;
    await mutateInvite(`/admin/api/invite-codes/${invite.publicId}/disable`, "POST", "Invite code disabled.");
  }

  async function deleteInvite(invite: InviteCode) {
    const confirmed = window.confirm(`Delete invite code ${invite.code}? It will be blocked for new signups and retained in audit history.`);
    if (!confirmed) return;
    await mutateInvite(`/admin/api/invite-codes/${invite.publicId}`, "DELETE", "Invite code deleted.");
  }

  async function mutateInvite(path: string, method: "POST" | "DELETE", success: string) {
    setSaving(true);
    try {
      await apiFetch(path, { method, csrfToken });
      setNotice(success);
      await loadInviteCodes();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  async function copyCode(invite: InviteCode) {
    await navigator.clipboard.writeText(invite.code);
    setCopiedCode(invite.code);
    window.setTimeout(() => setCopiedCode(""), 1400);
  }

  function exportCsv() {
    const header = ["code", "label", "status", "uses", "max_uses", "profit_share_percent", "wallet", "total_profit", "total_payout"];
    const lines = codes.map((invite) =>
      [
        invite.code,
        invite.label,
        invite.status,
        invite.currentUses,
        invite.maxUses || "unlimited",
        invite.profitSharePercent,
        invite.profitShareWallet,
        invite.totalInviteeProfit,
        invite.totalPaidToWallet
      ]
        .map((value) => `"${String(value).replaceAll('"', '""')}"`)
        .join(",")
    );
    const blob = new Blob([[header.join(","), ...lines].join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "invite-codes.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="ios-safe-page mx-auto min-h-screen w-full max-w-7xl">
      <header className="sticky top-0 z-20 -mx-4 border-b border-white/10 bg-[#070807]/90 px-4 py-3 backdrop-blur md:static md:mx-0 md:border-0 md:bg-transparent md:px-0">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase text-amber-300">Admin</p>
            <h1 className="text-2xl font-semibold text-white md:text-4xl">Invite Codes</h1>
          </div>
          <button
            className="tap-target inline-flex items-center justify-center rounded-full border border-white/10 bg-white/5 px-3 text-sm font-semibold text-white"
            onClick={() => void loadInviteCodes()}
            disabled={loading}
            type="button"
            aria-label="Refresh invite codes"
          >
            <RefreshCw className={`h-5 w-5 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </header>

      {!online && (
        <StateBanner tone="warning" icon={<AlertTriangle className="h-5 w-5" />} title="Offline">
          Changes are paused until your connection returns.
        </StateBanner>
      )}
      {notice && (
        <StateBanner tone="success" icon={<CheckCircle2 className="h-5 w-5" />} title="Saved">
          {notice}
        </StateBanner>
      )}
      {error && (
        <StateBanner tone="danger" icon={<AlertTriangle className="h-5 w-5" />} title="Needs review">
          {error}
        </StateBanner>
      )}

      <section className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Metric label="Active" value={formatNumber(summary.activeCodes)} />
        <Metric label="Uses" value={formatNumber(summary.totalUses)} />
        <Metric label="Profit" value={formatCurrency(summary.totalInviteeProfit)} />
        <Metric label="Paid to sufyanh" value={formatCurrency(summary.totalPaidToWallet)} />
        <Metric label="Failed" value={formatNumber(summary.failedPayouts)} danger={summary.failedPayouts > 0} />
      </section>

      <section className="mt-5 space-y-3">
        <div className="flex gap-2 overflow-x-auto pb-1">
          {statuses.map((item) => (
            <button
              key={item.key}
              className={`tap-target shrink-0 rounded-full px-4 text-sm font-semibold ${
                status === item.key ? "bg-amber-300 text-black" : "border border-white/10 bg-white/5 text-white"
              }`}
              onClick={() => setStatus(item.key)}
              type="button"
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className="grid gap-3 md:grid-cols-[1fr_220px]">
          <label className="relative block">
            <Search className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-white/45" />
            <input
              className="tap-target w-full rounded-2xl border border-white/10 bg-white/5 px-12 text-white placeholder:text-white/40"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search code, label, wallet, creator"
              type="search"
            />
          </label>
          <label className="relative block">
            <select
              className="tap-target w-full appearance-none rounded-2xl border border-white/10 bg-white/5 px-4 text-white"
              value={sort}
              onChange={(event) => setSort(event.target.value)}
            >
              {sortOptions.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-4 top-1/2 h-5 w-5 -translate-y-1/2 text-white/50" />
          </label>
        </div>
      </section>

      <section className="mt-5">
        {loading ? (
          <LoadingState />
        ) : codes.length === 0 ? (
          <EmptyState onCreate={() => startCreate("create")} />
        ) : (
          <>
            <div className="grid gap-3 lg:hidden">
              {codes.map((invite) => (
                <InviteCard
                  key={invite.publicId}
                  invite={invite}
                  copied={copiedCode === invite.code}
                  onCopy={() => void copyCode(invite)}
                  onEdit={() => startEdit(invite)}
                  onDetails={() => void openDetails(invite)}
                  onDisable={() => void disableInvite(invite)}
                  onDelete={() => void deleteInvite(invite)}
                />
              ))}
            </div>
            <DesktopTable
              codes={codes}
              copiedCode={copiedCode}
              onCopy={(invite) => void copyCode(invite)}
              onEdit={startEdit}
              onDetails={(invite) => void openDetails(invite)}
              onDisable={(invite) => void disableInvite(invite)}
              onDelete={(invite) => void deleteInvite(invite)}
            />
          </>
        )}
      </section>

      <div className="ios-bottom-bar fixed inset-x-0 bottom-0 z-30 border-t border-white/10 bg-[#070807]/95 pt-3 backdrop-blur">
        <div className="mx-auto grid max-w-7xl grid-cols-3 gap-2">
          <button className="tap-target rounded-2xl bg-amber-300 px-3 text-sm font-bold text-black" onClick={() => startCreate("create")} type="button">
            <Plus className="mx-auto h-5 w-5" />
            Create
          </button>
          <button className="tap-target rounded-2xl border border-white/10 bg-white/5 px-3 text-sm font-bold text-white" onClick={() => startCreate("batch")} type="button">
            <Clipboard className="mx-auto h-5 w-5" />
            Batch
          </button>
          <button className="tap-target rounded-2xl border border-white/10 bg-white/5 px-3 text-sm font-bold text-white" onClick={exportCsv} type="button">
            <Download className="mx-auto h-5 w-5" />
            Export
          </button>
        </div>
      </div>

      {sheet !== "closed" && (
        <InviteSheet
          mode={sheet}
          form={form}
          saving={saving}
          editing={editing}
          onClose={() => setSheet("closed")}
          onSubmit={submitForm}
          onChange={setForm}
        />
      )}

      {details && (
        <DetailsSheet
          invite={details}
          activeTab={detailTab}
          data={detailData}
          onTab={setDetailTab}
          onClose={() => setDetails(null)}
        />
      )}
    </main>
  );
}

function Metric({ label, value, danger = false }: { label: string; value: string; danger?: boolean }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.06] p-4">
      <p className="text-xs font-semibold uppercase text-white/50">{label}</p>
      <strong className={`mt-2 block text-xl text-white ${danger ? "text-red-300" : ""}`}>{value}</strong>
    </div>
  );
}

function StateBanner({ tone, icon, title, children }: { tone: "success" | "warning" | "danger"; icon: ReactNode; title: string; children: ReactNode }) {
  const tones = {
    success: "border-emerald-400/30 bg-emerald-400/10 text-emerald-100",
    warning: "border-amber-300/30 bg-amber-300/10 text-amber-100",
    danger: "border-red-400/30 bg-red-400/10 text-red-100"
  };
  return (
    <section className={`mt-4 flex gap-3 rounded-2xl border p-4 ${tones[tone]}`}>
      <div className="pt-0.5">{icon}</div>
      <div>
        <strong className="block text-sm">{title}</strong>
        <p className="text-sm opacity-85">{children}</p>
      </div>
    </section>
  );
}

function LoadingState() {
  return (
    <div className="grid gap-3">
      {[0, 1, 2].map((item) => (
        <div key={item} className="h-36 animate-pulse rounded-2xl border border-white/10 bg-white/[0.06]" />
      ))}
    </div>
  );
}

function EmptyState({ onCreate }: { onCreate: () => void }) {
  return (
    <div className="rounded-3xl border border-dashed border-white/15 bg-white/[0.04] p-8 text-center">
      <ShieldCheck className="mx-auto h-10 w-10 text-amber-300" />
      <h2 className="mt-4 text-xl font-semibold text-white">No invite codes yet</h2>
      <p className="mx-auto mt-2 max-w-sm text-sm text-white/60">No invite codes yet. Create one code or batch-generate a set for a campaign.</p>
      <button className="tap-target mt-5 rounded-2xl bg-amber-300 px-5 font-bold text-black" onClick={onCreate} type="button">
        Create Code
      </button>
    </div>
  );
}

function InviteCard(props: {
  invite: InviteCode;
  copied: boolean;
  onCopy: () => void;
  onEdit: () => void;
  onDetails: () => void;
  onDisable: () => void;
  onDelete: () => void;
}) {
  const { invite } = props;
  return (
    <article className="rounded-2xl border border-white/10 bg-white/[0.06] p-4">
      <div className="flex items-start justify-between gap-3">
        <button className="min-w-0 text-left" onClick={props.onDetails} type="button">
          <div className="flex items-center gap-2">
            <code className="rounded-lg bg-black/30 px-2 py-1 font-mono text-base text-white">{invite.code}</code>
            <StatusBadge status={invite.status} />
          </div>
          <p className="mt-2 truncate text-sm text-white/60">{invite.label || "No label"}</p>
        </button>
        <ActionMenu {...props} />
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
        <MiniStat label="Uses" value={`${invite.currentUses}${invite.maxUses ? `/${invite.maxUses}` : ""}`} />
        <MiniStat label="Share" value={`${invite.profitSharePercent}%`} />
        <MiniStat label="Profit" value={formatCurrency(invite.totalInviteeProfit)} />
        <MiniStat label="Paid" value={formatCurrency(invite.totalPaidToWallet)} />
      </div>
      <div className="mt-4 flex items-center justify-between rounded-xl bg-black/20 px-3 py-2 text-sm text-white/70">
        <span className="inline-flex items-center gap-2">
          <Wallet className="h-4 w-4 text-emerald-300" />
          {invite.profitShareWallet}
        </span>
        <button className="tap-target rounded-xl px-3 font-semibold text-amber-200" onClick={props.onCopy} type="button">
          {props.copied ? "Copied" : "Copy"}
        </button>
      </div>
    </article>
  );
}

function DesktopTable({
  codes,
  copiedCode,
  onCopy,
  onEdit,
  onDetails,
  onDisable,
  onDelete
}: {
  codes: InviteCode[];
  copiedCode: string;
  onCopy: (invite: InviteCode) => void;
  onEdit: (invite: InviteCode) => void;
  onDetails: (invite: InviteCode) => void;
  onDisable: (invite: InviteCode) => void;
  onDelete: (invite: InviteCode) => void;
}) {
  return (
    <div className="hidden overflow-hidden rounded-2xl border border-white/10 bg-white/[0.05] lg:block">
      <table className="w-full text-left text-sm">
        <thead className="border-b border-white/10 text-xs uppercase text-white/50">
          <tr>
            <th className="px-4 py-3">Code</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3">Uses</th>
            <th className="px-4 py-3">Profit</th>
            <th className="px-4 py-3">Paid</th>
            <th className="px-4 py-3">Wallet</th>
            <th className="px-4 py-3">Expires</th>
            <th className="px-4 py-3"></th>
          </tr>
        </thead>
        <tbody>
          {codes.map((invite) => (
            <tr key={invite.publicId} className="border-b border-white/5">
              <td className="px-4 py-4">
                <button className="text-left" onClick={() => onDetails(invite)} type="button">
                  <code className="font-mono text-white">{invite.code}</code>
                  <span className="block text-white/50">{invite.label || "No label"}</span>
                </button>
              </td>
              <td className="px-4 py-4"><StatusBadge status={invite.status} /></td>
              <td className="px-4 py-4 text-white/80">{invite.currentUses}{invite.maxUses ? `/${invite.maxUses}` : ""}</td>
              <td className="px-4 py-4 text-white/80">{formatCurrency(invite.totalInviteeProfit)}</td>
              <td className="px-4 py-4 text-white/80">{formatCurrency(invite.totalPaidToWallet)}</td>
              <td className="px-4 py-4 text-white/80">{invite.profitShareWallet}</td>
              <td className="px-4 py-4 text-white/80">{formatDate(invite.expiresAt)}</td>
              <td className="px-4 py-4">
                <div className="flex justify-end gap-2">
                  <IconButton label={copiedCode === invite.code ? "Copied" : "Copy"} onClick={() => onCopy(invite)} icon={<Copy className="h-4 w-4" />} />
                  <IconButton label="Edit" onClick={() => onEdit(invite)} icon={<Edit3 className="h-4 w-4" />} />
                  <IconButton label="Disable" onClick={() => onDisable(invite)} icon={<AlertTriangle className="h-4 w-4" />} />
                  <IconButton label="Delete" onClick={() => onDelete(invite)} icon={<Trash2 className="h-4 w-4" />} danger />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ActionMenu(props: {
  invite: InviteCode;
  copied: boolean;
  onCopy: () => void;
  onEdit: () => void;
  onDetails: () => void;
  onDisable: () => void;
  onDelete: () => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button className="tap-target rounded-full border border-white/10 bg-white/5 text-white" onClick={() => setOpen((value) => !value)} type="button" aria-label="Invite actions">
        <MoreHorizontal className="mx-auto h-5 w-5" />
      </button>
      {open && (
        <div className="absolute right-0 top-12 z-10 w-44 rounded-2xl border border-white/10 bg-[#111514] p-2 shadow-2xl">
          <MenuButton icon={<Copy className="h-4 w-4" />} label={props.copied ? "Copied" : "Copy"} onClick={props.onCopy} />
          <MenuButton icon={<Edit3 className="h-4 w-4" />} label="Edit" onClick={props.onEdit} />
          <MenuButton icon={<AlertTriangle className="h-4 w-4" />} label="Disable" onClick={props.onDisable} />
          <MenuButton icon={<Trash2 className="h-4 w-4" />} label="Delete" onClick={props.onDelete} danger />
        </div>
      )}
    </div>
  );
}

function InviteSheet({
  mode,
  form,
  saving,
  editing,
  onClose,
  onSubmit,
  onChange
}: {
  mode: "create" | "batch" | "edit";
  form: InviteFormPayload;
  saving: boolean;
  editing: InviteCode | null;
  onClose: () => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onChange: (form: InviteFormPayload) => void;
}) {
  const title = mode === "edit" ? `Edit ${editing?.code}` : mode === "batch" ? "Batch Generate" : "Create Invite Code";
  return (
    <div className="fixed inset-0 z-40 bg-black/60" role="dialog" aria-modal="true">
      <div className="absolute inset-x-0 bottom-0 rounded-t-3xl border border-white/10 bg-[#0c0f0e] shadow-2xl md:left-1/2 md:top-1/2 md:bottom-auto md:max-w-2xl md:-translate-x-1/2 md:-translate-y-1/2 md:rounded-3xl">
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <h2 className="text-lg font-semibold text-white">{title}</h2>
          <button className="tap-target rounded-full border border-white/10 bg-white/5" onClick={onClose} type="button" aria-label="Close form">
            <X className="mx-auto h-5 w-5" />
          </button>
        </div>
        <form className="sheet-scroll overflow-y-auto px-5 py-4" onSubmit={onSubmit}>
          <div className="grid gap-4 md:grid-cols-2">
            {mode !== "batch" ? (
              <Field label="Invite code">
                <input className="Input" value={form.code || ""} onChange={(event) => onChange({ ...form, code: event.target.value })} placeholder="SUFYAN20" />
              </Field>
            ) : (
              <>
                <Field label="Code prefix">
                  <input className="Input" value={form.codePrefix || ""} onChange={(event) => onChange({ ...form, codePrefix: event.target.value })} placeholder="SUFYAN" />
                </Field>
                <Field label="Batch count">
                  <input className="Input" type="number" min={1} max={100} value={form.batchCount || 1} onChange={(event) => onChange({ ...form, batchCount: Number(event.target.value) })} />
                </Field>
              </>
            )}
            <Field label="Label">
              <input className="Input" value={form.label || ""} onChange={(event) => onChange({ ...form, label: event.target.value })} placeholder="Partner campaign" />
            </Field>
            <Field label="Expiration date">
              <input className="Input" type="date" value={form.expirationDate || ""} onChange={(event) => onChange({ ...form, expirationDate: event.target.value })} />
            </Field>
            <Field label="Max uses">
              <input className="Input" type="number" min={1} inputMode="numeric" value={form.maxUses ?? ""} onChange={(event) => onChange({ ...form, maxUses: event.target.value ? Number(event.target.value) : "" })} placeholder="Unlimited" />
            </Field>
            <Field label="Assigned role">
              <select className="Input" value={form.assignedRole || "user"} onChange={(event) => onChange({ ...form, assignedRole: event.target.value })}>
                <option value="user">User</option>
                <option value="premium">Premium</option>
                <option value="trader">Trader</option>
                <option value="viewer">Viewer</option>
              </select>
            </Field>
            <Field label="Profit-share percentage">
              <input className="Input" type="number" min={0} max={100} step="0.01" inputMode="decimal" value={form.profitSharePercent} onChange={(event) => onChange({ ...form, profitSharePercent: Number(event.target.value) })} />
            </Field>
            <Field label="Destination wallet">
              <input className="Input" value={form.profitShareWallet} onChange={(event) => onChange({ ...form, profitShareWallet: event.target.value })} />
            </Field>
            <Field label="Start date">
              <input className="Input" type="datetime-local" value={form.profitShareStartsAt || ""} onChange={(event) => onChange({ ...form, profitShareStartsAt: event.target.value })} />
            </Field>
            <Field label="End date">
              <input className="Input" type="datetime-local" value={form.profitShareEndsAt || ""} onChange={(event) => onChange({ ...form, profitShareEndsAt: event.target.value })} />
            </Field>
          </div>
          <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-sm font-semibold text-white">Vault scope</p>
            <div className="mt-3 grid grid-cols-2 gap-2">
              {vaultTypeOptions.map((vaultType) => (
                <label key={vaultType} className="tap-target flex items-center gap-2 rounded-xl border border-white/10 bg-black/20 px-3 text-sm text-white">
                  <input
                    type="checkbox"
                    checked={form.appliesToVaultTypes.includes(vaultType)}
                    onChange={(event) => {
                      const next = event.target.checked
                        ? [...form.appliesToVaultTypes, vaultType]
                        : form.appliesToVaultTypes.filter((item) => item !== vaultType);
                      onChange({ ...form, appliesToVaultTypes: next });
                    }}
                  />
                  {vaultType}
                </label>
              ))}
            </div>
          </div>
          <div className="mt-4 grid gap-2 md:grid-cols-2">
            <Toggle label="Profit share active" checked={form.profitShareActive} onChange={(checked) => onChange({ ...form, profitShareActive: checked })} />
            <Toggle label="Invite code active" checked={form.isActive} onChange={(checked) => onChange({ ...form, isActive: checked })} />
          </div>
          <div className="sticky bottom-0 -mx-5 mt-5 border-t border-white/10 bg-[#0c0f0e] px-5 py-4">
            <button className="tap-target w-full rounded-2xl bg-amber-300 font-bold text-black disabled:opacity-60" type="submit" disabled={saving}>
              {saving ? <Loader2 className="mx-auto h-5 w-5 animate-spin" /> : mode === "edit" ? "Save Changes" : "Create"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function DetailsSheet({
  invite,
  activeTab,
  data,
  onTab,
  onClose
}: {
  invite: InviteCode;
  activeTab: "overview" | "invitees" | "payouts" | "audit";
  data: { usages: Usage[]; payouts: Payout[]; auditLogs: AuditLog[] };
  onTab: (tab: "overview" | "invitees" | "payouts" | "audit") => void;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-40 bg-black/60" role="dialog" aria-modal="true">
      <div className="absolute inset-x-0 bottom-0 rounded-t-3xl border border-white/10 bg-[#0c0f0e] shadow-2xl md:left-1/2 md:top-1/2 md:bottom-auto md:max-w-3xl md:-translate-x-1/2 md:-translate-y-1/2 md:rounded-3xl">
        <div className="flex items-start justify-between gap-3 border-b border-white/10 px-5 py-4">
          <div>
            <code className="font-mono text-xl text-white">{invite.code}</code>
            <p className="mt-1 text-sm text-white/60">{invite.label || "No label"}</p>
          </div>
          <button className="tap-target rounded-full border border-white/10 bg-white/5" onClick={onClose} type="button" aria-label="Close details">
            <X className="mx-auto h-5 w-5" />
          </button>
        </div>
        <div className="flex gap-2 overflow-x-auto border-b border-white/10 px-5 py-3">
          {(["overview", "invitees", "payouts", "audit"] as const).map((tab) => (
            <button key={tab} className={`tap-target rounded-full px-4 text-sm font-semibold ${activeTab === tab ? "bg-white text-black" : "bg-white/5 text-white"}`} onClick={() => onTab(tab)} type="button">
              {tab[0].toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
        <div className="sheet-scroll overflow-y-auto p-5">
          {activeTab === "overview" && (
            <div className="grid gap-3 md:grid-cols-2">
              <MiniStat label="Status" value={invite.status.replace("_", " ")} />
              <MiniStat label="Destination" value={invite.profitShareWallet} />
              <MiniStat label="Share" value={`${invite.profitSharePercent}%`} />
              <MiniStat label="Paid" value={formatCurrency(invite.totalPaidToWallet)} />
              <MiniStat label="Profit" value={formatCurrency(invite.totalInviteeProfit)} />
              <MiniStat label="Expiration" value={formatDate(invite.expiresAt)} />
            </div>
          )}
          {activeTab === "invitees" && <UsageList usages={data.usages} />}
          {activeTab === "payouts" && <PayoutList payouts={data.payouts} />}
          {activeTab === "audit" && <AuditList logs={data.auditLogs} />}
        </div>
      </div>
    </div>
  );
}

function UsageList({ usages }: { usages: Usage[] }) {
  if (!usages.length) return <p className="text-sm text-white/60">No invitees have joined with this code.</p>;
  return (
    <div className="grid gap-2">
      {usages.map((usage) => (
        <div key={usage.publicId} className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
          <strong className="text-white">{usage.invitee}</strong>
          <p className="text-sm text-white/55">{formatDate(usage.usedAt)} · {usage.acceptedDisclosureVersion}</p>
        </div>
      ))}
    </div>
  );
}

function PayoutList({ payouts }: { payouts: Payout[] }) {
  if (!payouts.length) return <p className="text-sm text-white/60">No profit-share payouts have been recorded.</p>;
  return (
    <div className="grid gap-2">
      {payouts.map((payout) => (
        <div key={payout.publicId} className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
          <div className="flex items-center justify-between gap-3">
            <strong className="text-white">{formatCurrency(payout.payoutAmount)}</strong>
            <span className={`rounded-full px-2 py-1 text-xs font-semibold ${payout.status === "completed" ? "bg-emerald-400/15 text-emerald-200" : "bg-red-400/15 text-red-200"}`}>
              {payout.status}
            </span>
          </div>
          <p className="mt-1 text-sm text-white/60">{payout.profitSharePercent}% from {formatCurrency(payout.sourceProfitAmount)} profit to {payout.destinationWallet}</p>
          {payout.failedReason && <p className="mt-2 text-sm text-red-200">Wallet credit failed. The payout was not completed and needs review.</p>}
        </div>
      ))}
    </div>
  );
}

function AuditList({ logs }: { logs: AuditLog[] }) {
  if (!logs.length) return <p className="text-sm text-white/60">No admin audit entries found.</p>;
  return (
    <div className="grid gap-2">
      {logs.map((log) => (
        <div key={log.publicId} className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
          <strong className="text-white">{log.action.replaceAll("_", " ")}</strong>
          <p className="text-sm text-white/55">{formatDate(log.createdAt)} · {log.admin || "system"} · {log.ipAddress}</p>
        </div>
      ))}
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="grid gap-2 text-sm font-semibold text-white/70">
      {label}
      {children}
    </label>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="tap-target flex items-center justify-between rounded-2xl border border-white/10 bg-white/[0.04] px-4 text-sm font-semibold text-white">
      {label}
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-black/20 p-3">
      <p className="text-xs uppercase text-white/45">{label}</p>
      <strong className="mt-1 block break-words text-white">{value}</strong>
    </div>
  );
}

function StatusBadge({ status }: { status: InviteStatus }) {
  const palette: Record<InviteStatus, string> = {
    active: "bg-emerald-400/15 text-emerald-200",
    disabled: "bg-zinc-400/15 text-zinc-200",
    expired: "bg-amber-300/15 text-amber-100",
    fully_used: "bg-sky-300/15 text-sky-100",
    deleted: "bg-red-400/15 text-red-100"
  };
  return <span className={`rounded-full px-2 py-1 text-xs font-semibold ${palette[status]}`}>{status.replace("_", " ")}</span>;
}

function IconButton({ label, icon, onClick, danger = false }: { label: string; icon: ReactNode; onClick: () => void; danger?: boolean }) {
  return (
    <button className={`tap-target rounded-xl border px-3 ${danger ? "border-red-400/30 bg-red-400/10 text-red-100" : "border-white/10 bg-white/5 text-white"}`} onClick={onClick} type="button" title={label} aria-label={label}>
      {icon}
    </button>
  );
}

function MenuButton({ icon, label, onClick, danger = false }: { icon: ReactNode; label: string; onClick: () => void; danger?: boolean }) {
  return (
    <button className={`tap-target flex w-full items-center gap-2 rounded-xl px-3 text-left text-sm font-semibold ${danger ? "text-red-100" : "text-white"}`} onClick={onClick} type="button">
      {icon}
      {label}
    </button>
  );
}

function normalizeFormPayload(form: InviteFormPayload, mode: "create" | "batch" | "edit"): InviteFormPayload {
  return {
    ...form,
    code: mode === "batch" ? undefined : form.code,
    batchCount: mode === "batch" ? Number(form.batchCount || 1) : 1,
    maxUses: form.maxUses === "" ? "" : Number(form.maxUses),
    profitSharePercent: Number(form.profitSharePercent || 0),
    profitShareWallet: form.profitShareWallet || "sufyanh",
    profitShareStartsAt: form.profitShareStartsAt || undefined,
    profitShareEndsAt: form.profitShareEndsAt || undefined,
    expirationDate: form.expirationDate || undefined
  };
}

function toDateInput(value: string | null) {
  if (!value) return "";
  return new Date(value).toISOString().slice(0, 10);
}

function toDateTimeInput(value: string | null) {
  if (!value) return "";
  return new Date(value).toISOString().slice(0, 16);
}

function errorMessage(err: unknown) {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "The request could not be completed.";
}
