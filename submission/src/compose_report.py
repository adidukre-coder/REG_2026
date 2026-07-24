"""
REG2026 — final pathology report composer (the 28%-weighted answer).

The report is a deterministic-ish function of the predicted diagnostic attributes:
  "{Organ}, {procedure lowercased};" + numbered (or single) diagnosis list,
  with Nottingham sub-scores appended for invasive carcinoma.
Note: GT reports use the literal two-char sequence "\\n" (backslash-n), not real newlines.

First-pass rule-based composer. Covers the dominant pattern; rare lesion-specific
sub-formats (e.g. DCIS Type/Nuclear grade/Necrosis lines) are TODO refinements — or
defer report phrasing to a judge-aligned Qwen later. compose_report(answers) -> str.
"""
from __future__ import annotations

NL = "\\n"  # GT uses literal backslash-n

# When the traversal's predicted edges skip straight from a neoplasm-workup question to the
# final-report question (short-circuiting "additional finding" -> "number of diagnoses" ->
# "#1 diagnosis"), pred_active never gets a "#N diagnosis" key and the composed report would be
# header-only. Falling back to the last real diagnostic attribute we DO have recovers most of the
# lost report score at zero risk: this only fires when dxs is already empty, so it can only add
# content, never change/remove an existing composed diagnosis.
_FALLBACK_DX_KEYS = ["What is the histologic type of neoplasm?"]


def _grading_detail(dx: str, a: dict, indent: str = "  ") -> str:
    # breast invasive carcinoma -> append Nottingham sub-scores if available
    if "nvasive carcinoma of no special type" in dx or "nvasive breast carcinoma" in dx:
        t = a.get("What is the score for tubular differentiation?")
        n = a.get("What is the score for nuclear pleomorphism?")
        m = a.get("What is the score for mitotic rate?")
        if t and n and m:
            return f" (Tubule formation: {t}, Nuclear grade: {n}, Mitoses: {m})"
        return ""
    # breast DCIS -> append structured Type/Nuclear grade/Necrosis sub-lines if available
    # (mapping confirmed against real GT trace: architectural pattern -> Type, nuclear grade of
    # lesion -> Nuclear grade, necrosis y/n -> Present/Absent). Each line only added if predicted.
    # `indent` must match the caller's per-diagnosis indent (2sp for a single diagnosis, wider
    # aligned-under-the-number indent for a numbered list) - GT indents these sub-lines to
    # align under the diagnosis text either way, confirmed against both single- and
    # numbered-diagnosis GT DCIS traces.
    if "ductal carcinoma in situ" in dx.lower():
        lines = []
        typ = a.get("What is the architectural pattern of lesion?")
        if typ:
            lines.append(f"{indent}- Type: {typ}")
        ng = a.get("What is the nuclear grade of lesion?")
        if ng:
            lines.append(f"{indent}- Nuclear grade: {ng}")
        necrosis_ans = a.get("Is there any necrosis present?")
        if necrosis_ans:
            necrosis = "Present" if necrosis_ans.lower().startswith("yes") else "Absent"
            if necrosis == "Present":
                necrosis += _necrosis_qualifier(a.get("What is the type of necrosis?", ""))
            lines.append(f"{indent}- Necrosis: {necrosis}")
        if lines:
            return NL + NL.join(lines)
    return ""


def _necrosis_qualifier(necrosis_type: str) -> str:
    # "What is the type of necrosis?" head has 3 labels: "Focal necrosis", "Comedo-type
    # necrosis", "Necrosis" (generic). GT strips only the trailing " necrosis" and keeps
    # the rest verbatim - confirmed against real GT: "Focal necrosis" -> "(Focal)" AND
    # "Comedo-type necrosis" -> "(Comedo-type)" (the "-type" stays; an earlier version of
    # this also stripped "-type", which was wrong - GT never drops it). Generic "Necrosis"
    # carries no extra info, so it adds nothing.
    t = necrosis_type.strip()
    if not t or t.lower() == "necrosis":
        return ""
    if t.lower().endswith(" necrosis"):
        return f" ({t[:-len(' necrosis')]})"
    return ""


_ULCERATION_TEXT = {
    "Confined to the mucosa": "erosion",
    "Extend through the muscularis mucosae": "ulcer",
}


def _lower_first(s: str) -> str:
    # "Reactive atypia" -> "reactive atypia" but "CMV-infected cells" stays as-is
    # (acronym first word). Matches the casing GT actually uses in both real examples.
    if not s:
        return s
    first_word = s.split(" ", 1)[0].split("-", 1)[0]
    if first_word.isupper() and len(first_word) > 1:
        return s
    return s[0].lower() + s[1:]


def _gastritis_text(dx: str, a: dict) -> str:
    # Stomach "Chronic gastritis" reports have their own template, entirely separate from
    # the diagnosis-attribute pipeline everything else here uses: "Is there any active
    # inflammation present?" modifies the DIAGNOSIS WORD ITSELF ("Chronic gastritis" ->
    # "Chronic active gastritis"), then a checklist of other findings (ulceration,
    # intestinal metaplasia, foveolar epithelial hyperplasia, lymphoid aggregate/follicle,
    # a rare free-text "additional finding") is appended as "with X" (1 finding) or a
    # numbered "1) X\n       2) Y" list (2+, fixed 7-space continuation indent, confirmed
    # against every GT numbered example in train_labels.json). Ulceration's exact wording
    # (erosion vs ulcer) depends on "What is the depth of ulceration?" (confirmed 100%
    # consistent across all matching GT records: confined-to-mucosa -> erosion,
    # extends-through-muscularis-mucosae -> ulcer). When BOTH ulceration and lymphoid
    # aggregate are present, GT drops the "with " prefix on the numbered list - a genuine
    # quirk, confirmed consistent across all 30 GT numbered-list gastritis cases (24 with,
    # 6 without, and the "without" set is exactly {ulceration present AND lymphoid
    # aggregate present}, no exceptions).
    diagnosis = "Chronic active gastritis" if a.get(
        "Is there any active inflammation present?", "").startswith("Yes") else dx

    has_ulceration = a.get("Is there any ulceration present?", "").startswith("Yes")
    has_la = a.get("Is there any lymphoid aggregate present?", "").startswith("Yes")
    findings = []
    if has_ulceration:
        findings.append(_ULCERATION_TEXT.get(a.get("What is the depth of ulceration?", ""), "ulcer"))
    if a.get("Is there any intestinal metaplasia present?", "").startswith("Yes"):
        findings.append("intestinal metaplasia")
    if a.get("Is there any foveolar epithelial hyperplasia present?", "").startswith("Yes"):
        findings.append("foveolar epithelial hyperplasia")
    if a.get("Is there any lymphoid follicle present?", "").startswith("Yes"):
        findings.append("lymphoid follicle")  # supersedes lymphoid aggregate when both present
    elif has_la:
        findings.append("lymphoid aggregate")
    extra = a.get("What is the additional finding?")
    if extra:
        findings.append(_lower_first(extra))

    if not findings:
        return diagnosis
    if len(findings) == 1:
        return f"{diagnosis} with {findings[0]}"
    omit_with = has_ulceration and has_la
    lines = [f"  {'' if omit_with else 'with '}1) {findings[0]}"]
    lines += [f"       {i}) {f}" for i, f in enumerate(findings[1:], 2)]
    return diagnosis + NL + NL.join(lines)


def _expand_diagnosis(dx: str, a: dict) -> str:
    # "Micro-invasive carcinoma" is the raw categorical label, but GT always renders it as
    # one word "Microinvasive carcinoma" in prose (confirmed 7/7 occurrences in
    # train_labels.json, breast only). Plain text substitution, can't misfire elsewhere.
    dx = dx.replace("Micro-invasive", "Microinvasive")
    # Colon/stomach adenomas: append "with {grade} dysplasia" when the model predicted
    # dysplasia + a grade for it (confirmed 100% consistent across all 1183 matching GT
    # records in train_labels.json, colon+stomach only - the 4 "exceptions" found are
    # diagnosis labels that already have the dysplasia clause baked into the categorical
    # value itself, e.g. "Sessile serrated lesion with low grade dysplasia", hence the
    # "dysplasia" in dx.lower() guard). Text-only match, no other organ has this attribute.
    if "gastritis" in dx.lower():
        return _gastritis_text(dx, a)
    if "dysplasia" not in dx.lower() and a.get("Is there any dysplasia present?", "").startswith("Yes"):
        grade = a.get("What is the grade of dysplasia?")
        if grade:
            return f"{dx} with {grade.lower()} dysplasia"
    return dx


def _split_involvement(dx: str, indent: str) -> str:
    # Bladder invasive-urothelial-carcinoma diagnoses are a flat model-predicted string
    # ("...carcinoma with involvement of X"), but GT reports always break the line right
    # before "with involvement of" (comma + newline + indent aligned under the diagnosis
    # text). Confirmed against all 9 GT bladder cases matching this pattern - always this
    # exact split, never anything else. Text-only, so it can only ever match this literal
    # marker; no interaction with other organs (checked: marker never appears outside bladder).
    marker = " with involvement of "
    if marker in dx:
        head, tail = dx.split(marker, 1)
        return f"{head},{NL}{indent}with involvement of {tail}"
    return dx


def _split_glandular(dx: str) -> str:
    # Cervix HSIL "...with glandular involvement" is likewise a flat categorical string,
    # but GT breaks it onto its own 2-space-indented line - confirmed 50/51 matching GT
    # records (colon+stomach/breast-style split). The lone exception is the only numbered
    # (2-diagnosis) case with this marker in the whole corpus, which stays inline instead -
    # so this split is applied only for the single-diagnosis path, never the numbered one.
    marker = " with glandular involvement"
    if dx.endswith(marker):
        return dx[: -len(marker)] + NL + "  with glandular involvement"
    return dx


def _split_malt(dx: str) -> str:
    # Colon/rectum "...marginal zone lymphoma of mucosa-associated lymphoid tissue (MALT
    # lymphoma)" is likewise flat, but GT always breaks it right before " of mucosa..." onto
    # its own 2-space-indented line - confirmed 4/4 matching GT records, no exceptions.
    marker = " of mucosa-associated lymphoid tissue"
    if marker in dx:
        head, tail = dx.split(marker, 1)
        return f"{head}{NL}  of mucosa-associated lymphoid tissue{tail}"
    return dx


_PROSTATE_SINGLE_SPACE_DX = {"Small cell carcinoma", "Malignant lymphoma"}


def _prostate_dx_text(dx: str, a: dict) -> str:
    """Single diagnosis body: '{dx}, Gleason's score {GS}, grade group N (Gleason pattern 4: P%), tumor volume: V'."""
    if "denocarcinoma" not in dx.lower():
        return dx  # benign / 'No tumor present' / other
    parts = [dx]
    gs = a.get("What is the Gleason score?")
    if gs:
        parts.append(f"Gleason's score {gs}")
    gg = a.get("What is the grade group?", "")          # 'Grade group 2' -> 'grade group 2'
    gg = gg.replace("Grade group", "grade group")
    p4 = a.get("What is the percentage of Gleason pattern 4?")
    if gg:
        parts.append(f"{gg} (Gleason pattern 4: {p4})" if p4 else gg)
    tv = a.get("What is the tumor volume?")
    if tv:
        parts.append(f"tumor volume: {tv}")
    return ", ".join(parts)


def _prostate_body(a: dict) -> str:
    dxs = []
    k = 1
    while f"What is the #{k} diagnosis?" in a:
        dxs.append(a[f"What is the #{k} diagnosis?"]); k += 1
    if not dxs:
        return ""
    if len(dxs) == 1:
        # Two rare diagnoses ("Small cell carcinoma", "Malignant lymphoma") use a single
        # leading space instead of the usual 2 - confirmed 100% consistent (3/3 occurrences
        # in train_labels.json), a genuine dataset quirk, not worth more than a literal match.
        indent = " " if dxs[0] in _PROSTATE_SINGLE_SPACE_DX else "  "
        return indent + _prostate_dx_text(dxs[0], a)
    # 2+ diagnoses (e.g. adenocarcinoma/no-tumor + a secondary inflammation finding) render as
    # a numbered list, same as every other organ - the original single-dx-only version silently
    # dropped the #2+ diagnosis entirely. Confirmed against every GT multi-dx prostate case.
    return NL.join(f"  {i}. {_prostate_dx_text(d, a)}" for i, d in enumerate(dxs, 1))


def compose_report(a: dict) -> str:
    organ = a.get("What is the organ?", "")
    proc = a.get("What is the procedure?", "")

    if organ == "Prostate":
        header = f"{organ}, {proc.lower()};"
        body = _prostate_body(a)
        return header + NL + body if body else header

    # gather #1..#k diagnoses
    dxs = []
    k = 1
    while f"What is the #{k} diagnosis?" in a:
        dxs.append(a[f"What is the #{k} diagnosis?"])
        k += 1

    # Breast "Intraductal papilloma" always renders as "core needle biopsy" in GT even when
    # the model predicted "Mammotome biopsy" - confirmed 7/7 occurrences, and confirmed NOT a
    # general Mammotome->core-needle rule (other diagnoses keep "mammotome biopsy" correctly
    # in 25/32 GT cases). A genuine diagnosis-conditional report-writing convention.
    if dxs and dxs[0] == "Intraductal papilloma" and proc == "Mammotome biopsy":
        proc = "Core needle biopsy"
    header = f"{organ}, {proc.lower()};"

    if not dxs:
        for fk in _FALLBACK_DX_KEYS:
            if a.get(fk):
                dxs = [a[fk]]
                break

    if not dxs:
        return header  # no diagnosis predicted

    if len(dxs) == 1:
        dx = _split_malt(_split_glandular(_split_involvement(_expand_diagnosis(dxs[0], a), "  ")))
        out = header + NL + "  " + dx + _grading_detail(dxs[0], a, "  ")
    else:
        def _line(i, d):
            prefix = f"{i}. "
            indent = "  " + " " * len(prefix)
            dx = _split_involvement(_expand_diagnosis(d, a), indent)
            return f"  {prefix}{dx}{_grading_detail(d, a, indent)}"
        body = NL.join(_line(i, d) for i, d in enumerate(dxs, 1))
        out = header + NL + body
    # Bladder TURP reports end with a muscle-proper note, preceded by a blank line (confirmed
    # constant "\n\nNote)" across all GT bladder cases). Driven by "Is there any muscularis
    # propria present?" - confirmed a PERFECT (100%, n=1079) predictor of includes/does-not-
    # include across the full labeled corpus; defaults to "includes" (majority class, ~74%)
    # only when the attribute wasn't predicted at all.
    if organ == "Urinary bladder":
        mp = a.get("Is there any muscularis propria present?", "")
        verb = "does not include" if mp.startswith("No") else "includes"
        out += NL + NL + f"Note) The specimen {verb} muscle proper."
    return out
