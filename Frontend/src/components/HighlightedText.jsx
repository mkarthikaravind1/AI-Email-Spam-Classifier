// Feature #1: shows the analysed text with
//  - a background per word: how much it pushed the Statistical ML model
//    towards spam (red) or ham (green), stronger colour = stronger push
//  - a blue underline under text matched by a regex rule (hover for the rule)
//  - a dashed outline around links found by the link analyser
export default function HighlightedText({ text, mlWords = [], regexSpans = [], links = [] }) {
  const maxAbs = Math.max(0.001, ...mlWords.map((w) => Math.abs(w.contribution)));

  // Per-character attributes, then group runs of identical attributes
  const chars = Array.from({ length: text.length }, () => ({ c: 0, rules: [] }));
  for (const w of mlWords) {
    for (let i = w.start; i < w.end && i < text.length; i++) chars[i].c = w.contribution;
  }
  for (const s of regexSpans) {
    for (let i = s.start; i < s.end && i < text.length; i++) {
      if (!chars[i].rules.includes(s.rule)) chars[i].rules.push(s.rule);
    }
  }
  const linkChars = new Set();
  for (const l of links) {
    for (let i = l.start; i < l.end && i < text.length; i++) linkChars.add(i);
  }

  const segments = [];
  for (let i = 0; i < text.length; i++) {
    const key = `${chars[i].c}|${chars[i].rules.join(",")}|${linkChars.has(i)}`;
    const last = segments[segments.length - 1];
    if (last && last.key === key) last.text += text[i];
    else segments.push({ key, text: text[i], c: chars[i].c, rules: chars[i].rules, link: linkChars.has(i) });
  }

  return (
    <div>
      <p className="whitespace-pre-wrap break-words leading-8 text-base">
        {segments.map((s, i) => {
          const strength = Math.min(1, Math.abs(s.c) / maxAbs);
          const bg = s.c > 0 ? `rgba(227, 73, 72, ${0.12 + 0.55 * strength})`
            : s.c < 0 ? `rgba(0, 131, 0, ${0.1 + 0.4 * strength})` : "transparent";
          const title = [
            s.c ? `ML: ${s.c > 0 ? "towards spam" : "towards ham"} (${s.c > 0 ? "+" : ""}${s.c.toFixed(2)} log-odds)` : "",
            s.rules.length ? `Regex rule: ${s.rules.join(", ")}` : "",
            s.link ? "Link (see link safety)" : "",
          ].filter(Boolean).join("\n");
          return (
            <span
              key={i}
              title={title || undefined}
              style={{ backgroundColor: bg }}
              className={`rounded-sm ${s.rules.length ? "border-b-2 border-[#2a78d6]" : ""} ${s.link ? "outline-dashed outline-1 outline-gray-500" : ""}`}
            >
              {s.text}
            </span>
          );
        })}
      </p>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 text-xs text-gray-600">
        <span><span className="inline-block w-3 h-3 rounded-sm align-middle mr-1" style={{ background: "rgba(227,73,72,0.6)" }} />pushed ML towards spam</span>
        <span><span className="inline-block w-3 h-3 rounded-sm align-middle mr-1" style={{ background: "rgba(0,131,0,0.45)" }} />pushed ML towards ham</span>
        <span><span className="inline-block w-4 border-b-2 border-[#2a78d6] align-middle mr-1" />matched a regex rule</span>
        <span>Hover a word for details</span>
      </div>
    </div>
  );
}
