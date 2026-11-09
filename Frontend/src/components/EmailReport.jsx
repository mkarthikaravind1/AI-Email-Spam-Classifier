// Feature #9: details and sender checks for an uploaded .eml file
const LEVEL = {
  high: { label: "High risk", icon: "⛔", cls: "bg-red-100 text-red-800 border-red-300" },
  medium: { label: "Suspicious", icon: "⚠️", cls: "bg-amber-100 text-amber-900 border-amber-300" },
  low: { label: "No red flags", icon: "✓", cls: "bg-green-50 text-green-800 border-green-300" },
};
const AUTH_STYLE = { pass: "bg-green-100 text-green-800", fail: "bg-red-100 text-red-800",
  softfail: "bg-amber-100 text-amber-900" };

function Row({ label, children }) {
  if (!children) return null;
  return (
    <div className="grid grid-cols-[6.5rem_1fr] gap-2 py-1 border-b last:border-0">
      <dt className="text-gray-500">{label}</dt>
      <dd className="break-words">{children}</dd>
    </div>
  );
}

export default function EmailReport({ email }) {
  const checks = email.sender_checks;
  const style = LEVEL[checks.level];
  const sender = email.from.name ? `${email.from.name} <${email.from.address}>` : email.from.address;

  return (
    <div className="space-y-4 text-sm">
      <dl>
        <Row label="File">{email.filename}</Row>
        <Row label="From">{sender}</Row>
        <Row label="Reply-To">{email.reply_to}</Row>
        <Row label="Subject">{email.subject}</Row>
        <Row label="Date">{email.date}</Row>
        <Row label="Body">{email.body_format}</Row>
        <Row label="Authentication">
          {Object.keys(email.auth).length === 0 ? (
            <span className="text-gray-500">not recorded</span>
          ) : (
            <span className="flex flex-wrap gap-1">
              {Object.entries(email.auth).map(([k, v]) => (
                <span key={k} className={`px-2 py-0.5 rounded-full text-xs font-semibold ${AUTH_STYLE[v] ?? "bg-gray-100 text-gray-700"}`}>
                  {k.toUpperCase()}: {v}
                </span>
              ))}
            </span>
          )}
        </Row>
        <Row label="Attachments">
          {email.attachments.length > 0 &&
            email.attachments.map((a) => `${a.filename} (${a.content_type}, ${a.size} bytes)`).join(", ")}
        </Row>
      </dl>

      <div className={`border rounded-lg p-3 ${style.cls}`}>
        <p className="font-semibold">{style.icon} Sender checks: {style.label} · {checks.risk}/100</p>
        {checks.findings.length > 0 && (
          <ul className="mt-2 list-disc pl-5 space-y-0.5">
            {checks.findings.map((f) => <li key={f.kind}>{f.message}</li>)}
          </ul>
        )}
        {checks.notes.map((n) => <p key={n} className="mt-2 text-xs opacity-80">{n}</p>)}
      </div>
      <p className="text-xs text-gray-500">
        Sender checks are shown alongside the verdict, not voted into it: the text models were trained on SMS
        messages, which have no headers to learn from.
      </p>
    </div>
  );
}
