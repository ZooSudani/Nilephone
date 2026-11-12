import model from "../model.json";

// ===== التنظيف: يطابق دالة clean في Python حرفياً =====
function clean(t) {
  t = String(t);
  t = t.replace(/[\u064B-\u0652\u0640]/g, "");
  t = t.replace(/[أإآ]/g, "ا");
  t = t.replace(/ى/g, "ي").replace(/ة/g, "ه");
  t = t.toLowerCase();
  t = t.replace(/[^\u0621-\u064A0-9a-z\s]/g, " ");
  t = t.replace(/\s+/g, " ").trim();
  return t;
}

// ===== كلمات مفردة + ثنائية (ngram_range=(1,2)) =====
function features(text) {
  const toks = clean(text).split(" ").filter(Boolean);
  const grams = [...toks];
  for (let i = 0; i < toks.length - 1; i++) grams.push(toks[i] + " " + toks[i + 1]);
  return grams;
}

export function predict(text) {
  const { classes, class_log_prior, terms } = model;
  const counts = new Map();
  for (const g of features(text)) {
    if (Object.prototype.hasOwnProperty.call(terms, g)) counts.set(g, (counts.get(g) || 0) + 1);
  }
  // TF-IDF ثم تطبيع L2
  const x = [];
  let norm = 0;
  for (const [g, c] of counts) {
    const v = c * terms[g][0];
    x.push([g, v]);
    norm += v * v;
  }
  norm = Math.sqrt(norm) || 1;

  const scores = class_log_prior.slice();
  for (const [g, v] of x) {
    const w = v / norm;
    for (let k = 0; k < classes.length; k++) scores[k] += w * terms[g][k + 1];
  }
  // softmax
  const m = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sum);
  const best = probs.indexOf(Math.max(...probs));
  return { label: classes[best], confidence: probs[best], probs, known_terms: counts.size };
}

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });

export async function onRequestPost({ request }) {
  let body;
  try { body = await request.json(); } catch { return json({ error: "JSON غير صالح" }, 400); }
  const text = (body.text || "").toString().trim();
  if (!text) return json({ error: "الرجاء كتابة رسالة" }, 400);
  if (text.length > 1000) return json({ error: "الرسالة طويلة (الحد 1000 حرف)" }, 400);
  const r = predict(text);
  return json({
    company: model.company,
    label: r.label,
    confidence: r.confidence,
    scores: Object.fromEntries(model.classes.map((c, i) => [c, r.probs[i]])),
    known_terms: r.known_terms,
  });
}

export async function onRequestGet() {
  return json({ company: model.company, classes: model.classes, usage: "POST /predict {\"text\": \"...\"}" });
}
