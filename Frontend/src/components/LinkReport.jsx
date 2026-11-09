// Feature #8: offline phishing-link analysis results
const LEVEL_STYLE = {
  high: { label: "High risk", icon: "⛔", cls: "bg-red-100 text-red-800 border-red-300" },
  medium: { label: "Suspicious", icon: "⚠️", cls: "bg-amber-100 text-amber-900 border-amber-300" },
  low: { label: "No red flags", icon: "✓", cls: "bg-green-50 text-green-800 border-green-300" },
};

export default function LinkReport({ links }) {
  if (!links || links.links.length === 0) {
    return <p className="text-gray-500">No links found in this message.</p>;
  }
  return (
    <div className="space-y-3">
      {links.links.map((link) => {
        const style = LEVEL_STYLE[link.level];
        return (
          <div key={link.url} className={`border rounded-lg p-3 ${style.cls}`}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <code className="break-all text-sm">{link.url}</code>
              <span className="font-semibold text-sm whitespace-nowrap">
                {style.icon} {style.label} · {link.risk}/100
              </span>
            </div>
            {link.findings.length > 0 && (
              <ul className="mt-2 text-sm list-disc pl-5 space-y-0.5">
                {link.findings.map((f) => (
                  <li key={f.kind}>{f.message}</li>
                ))}
              </ul>
            )}
          </div>
        );
      })}
      <p className="text-xs text-gray-500">
        Checked offline: nothing was opened or fetched. Short links can't be expanded without visiting them.
      </p>
    </div>
  );
}
