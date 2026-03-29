"""
risk_lexicon_es_ca.py
=====================
Spanish + Catalan suicide-risk and mental-health lexicon for use as
bag-of-features in the BERT+LDA ensemble.

Categories and their clinical rationale
----------------------------------------
1.  suicidal_ideation   – direct expressions of suicidal intent / planning
2.  hopelessness        – Beck Hopelessness Scale language; strong predictor
3.  defeat_entrapment   – Williams' Arrested Flight Model; feeling trapped
4.  social_isolation    – perceived burdensomeness / thwarted belonging
5.  emotional_pain      – unbearable psychological pain (psychache)
6.  negation_amplifiers – absolutist / nihilistic language (eRisk finding)
7.  temporal_finality   – "last time", "forever", farewell language
8.  escape_avoidance    – desire to disappear / stop existing
9.  self_harm           – non-suicidal self-injury markers
10. protective_factors  – INVERSE signal: social support, help-seeking
    (lower score → higher risk)

Sources
-------
- LIWC-22 Spanish (Pennebaker et al.)
- MentalRiskES shared task lexicon (IberLEF 2023-2025)
- Sierra & Bel-Enguix (2022) "Suicide Risk Factors: A Language Analysis
  Approach in Social Media", Language in Society
- eRisk / CLPsych absolutist word list (Al-Mosaiwi & Johnstone 2018)
- Catalan Institut d'Estudis Catalans (IEC) equivalents
- WHO mhGAP Spanish/Catalan crisis terminology
"""

# ---------------------------------------------------------------------------
# 1. SUICIDAL IDEATION (ES + CA)
# ---------------------------------------------------------------------------
SUICIDAL_IDEATION = {
    # Spanish
    "suicidio", "suicidarme", "suicidarme", "suicida", "suicidarse",
    "quitarme la vida", "quitarme la vida", "acabar con mi vida",
    "acabar con todo", "matarme", "matarse", "hacerme daño",
    "hacerse daño", "quiero morir", "deseo morir", "pensar en morir",
    "no quiero vivir", "no quiero seguir viviendo", "no quiero seguir aquí",
    "no quiero estar aquí", "he pensado en suicidarme", "plan de suicidio",
    "método para morir", "pastillas para morir", "pienso en matarme",
    "tengo un plan", "ya lo tengo planeado", "me voy a matar",
    "voy a hacerlo", "esta es la última vez", "me despido",
    # Catalan
    "suïcidi", "suïcidar-me", "suïcidar-se", "suïcida",
    "treure'm la vida", "acabar amb la meva vida", "acabar amb tot",
    "matar-me", "fer-me mal", "vull morir", "desitjo morir",
    "no vull viure", "no vull seguir aquí", "no vull estar aquí",
    "he pensat a suïcidar-me", "pla de suïcidi",
}

# ---------------------------------------------------------------------------
# 2. HOPELESSNESS (ES + CA) — Beck Hopelessness Scale language
# ---------------------------------------------------------------------------
HOPELESSNESS = {
    # Spanish
    "sin esperanza", "desesperanza", "desesperado", "desesperanza",
    "no hay salida", "sin salida", "no tiene sentido", "sin sentido",
    "no vale la pena", "nada va a mejorar", "nunca va a mejorar",
    "nunca va a cambiar", "no hay futuro", "sin futuro",
    "todo está perdido", "no hay remedio", "imposible", "inútil",
    "sin solución", "no tiene arreglo", "para qué", "para qué sirve",
    "nada importa", "qué más da", "da igual", "me da igual todo",
    "no me importa nada", "ya no me importa", "nada tiene sentido",
    "todo es inútil", "rendirse", "me rindo", "ya me rendí",
    # Catalan
    "sense esperança", "desesperança", "desesperanzat", "desesperada",
    "no hi ha sortida", "sense sortida", "no té sentit", "sense sentit",
    "no val la pena", "res no millorarà", "mai no millorarà",
    "no hi ha futur", "sense futur", "tot està perdut",
    "per a què serveix", "res no importa", "m'és igual tot",
    "ja no m'importa", "res no té sentit", "tot és inútil",
}

# ---------------------------------------------------------------------------
# 3. DEFEAT & ENTRAPMENT (ES + CA) — Williams' Arrested Flight Model
# ---------------------------------------------------------------------------
DEFEAT_ENTRAPMENT = {
    # Spanish
    "atrapado", "atrapada", "no puedo escapar", "sin escapatoria",
    "encerrado", "encerrada", "preso", "presa", "no hay salida",
    "sin salida posible", "derrotado", "derrotada", "fracasado",
    "fracasada", "he fallado", "soy un fracaso", "soy una fracasada",
    "soy un inútil", "soy una inútil", "no sirvo para nada",
    "no valgo nada", "no valgo para nada", "soy un estorbo",
    "no puedo más", "ya no puedo más", "no aguanto más",
    "no soporto más", "al límite", "llegué al límite",
    "ya no tengo fuerzas", "sin fuerzas", "agotado", "agotada",
    # Catalan
    "atrapat", "atrapada", "no puc escapar", "sense escapatòria",
    "tancat", "tancada", "presoner", "presonera", "no hi ha sortida",
    "derrotat", "derrotada", "fracassat", "fracassada",
    "he fallat", "sóc un fracàs", "no serveixo per a res",
    "no valc res", "sóc un destorb", "no puc més", "ja no puc més",
    "no aguanto més", "no suporto més", "al límit",
    "ja no tinc forces", "sense forces", "esgotat", "esgotada",
}

# ---------------------------------------------------------------------------
# 4. SOCIAL ISOLATION (ES + CA) — Thwarted belonging / perceived burden
# ---------------------------------------------------------------------------
SOCIAL_ISOLATION = {
    # Spanish
    "solo", "sola", "completamente solo", "completamente sola",
    "nadie me entiende", "nadie me escucha", "nadie me quiere",
    "nadie se preocupa", "nadie me importa", "no tengo a nadie",
    "estoy solo en el mundo", "soy una carga", "soy un peso",
    "molesto a todos", "fastidio a todos", "estarían mejor sin mí",
    "serían más felices sin mí", "me echarán de menos",
    "nadie me echará de menos", "no me echará de menos nadie",
    "sin amigos", "sin familia", "abandonado", "abandonada",
    "rechazado", "rechazada", "incomprendido", "incomprendida",
    "invisible", "aislado", "aislada",
    # Catalan
    "sol", "sola", "completament sol", "completament sola",
    "ningú no m'entén", "ningú no m'escolta", "ningú no m'estima",
    "ningú no se'n preocupa", "no tinc ningú", "estic sol al món",
    "sóc una càrrega", "sóc un pes", "molesto tothom",
    "estarien millor sense mi", "serien més feliços sense mi",
    "ningú no em trobarà a faltar", "sense amics", "sense família",
    "abandonat", "abandonada", "rebutjat", "rebutjada",
    "incomprès", "incompresa", "invisible", "aïllat", "aïllada",
}

# ---------------------------------------------------------------------------
# 5. EMOTIONAL PAIN (ES + CA) — Schneidman's psychache
# ---------------------------------------------------------------------------
EMOTIONAL_PAIN = {
    # Spanish
    "dolor insoportable", "dolor inaguantable", "no puedo soportar el dolor",
    "sufrimiento", "sufriendo", "sufro mucho", "tanto dolor",
    "demasiado dolor", "dolor constante", "agonía", "agonizando",
    "tortura", "torturado", "torturada", "martirio", "angustia",
    "angustiado", "angustiada", "desesperación", "vacío interior",
    "vacío existencial", "me siento vacío", "me siento vacía",
    "oscuridad", "oscuro por dentro", "hundido", "hundida",
    "aplastado", "aplastada", "destrozado", "destrozada",
    "roto", "rota", "quebrado", "quebrada", "herido", "herida",
    # Catalan
    "dolor insuportable", "no puc suportar el dolor",
    "patiment", "patint", "pateixo molt", "tant dolor",
    "massa dolor", "dolor constant", "agonia", "agonitzant",
    "tortura", "torturat", "torturada", "martiri", "angoixa",
    "angoixat", "angoixada", "desesperació", "buit interior",
    "em sento buit", "em sento buida", "foscor", "fosc per dins",
    "enfonsat", "enfonsada", "destrossat", "destrossada",
    "trencat", "trencada", "ferit", "ferida",
}

# ---------------------------------------------------------------------------
# 6. NEGATION AMPLIFIERS / ABSOLUTIST LANGUAGE (ES + CA)
# — Al-Mosaiwi & Johnstone (2018): absolutist words elevated in
#   depression/suicidality forum posts vs. anxiety-only posts
# ---------------------------------------------------------------------------
NEGATION_AMPLIFIERS = {
    # Spanish
    "nunca", "jamás", "siempre", "todo", "nada", "nadie",
    "ninguno", "ninguna", "ni siquiera", "absolutamente nada",
    "completamente", "totalmente", "para siempre", "definitivamente",
    "imposible", "incapaz", "en absoluto", "de ninguna manera",
    "jamás podré", "nunca podré", "nunca lo lograré",
    "siempre fallo", "siempre fracaso", "todo está mal",
    "nada está bien", "nada saldrá bien",
    # Catalan
    "mai", "sempre", "tot", "res", "ningú",
    "cap", "ni tan sols", "absolutament res",
    "completament", "totalment", "per sempre", "definitivament",
    "impossible", "incapaç", "en absolut", "de cap manera",
    "mai podré", "mai ho aconseguiré", "sempre fallo",
    "sempre fracasso", "tot va malament", "res va bé",
}

# ---------------------------------------------------------------------------
# 7. TEMPORAL FINALITY (ES + CA) — farewell, "last time" language
# ---------------------------------------------------------------------------
TEMPORAL_FINALITY = {
    # Spanish
    "última vez", "por última vez", "es el final", "se acabó",
    "ya no habrá más", "fin de todo", "adiós para siempre",
    "me despido", "me estoy despidiendo", "despedida",
    "esta es mi despedida", "no volveré", "no regresaré",
    "ya no estaré", "ya no estaré aquí", "ya no existiré",
    "mañana no estaré", "pronto todo habrá terminado",
    "he regalado mis cosas", "he regalado todo",
    "ya me despedí de mis amigos", "ya me despedí de mi familia",
    # Catalan
    "última vegada", "per última vegada", "és el final", "s'ha acabat",
    "ja no hi haurà més", "fi de tot", "adéu per sempre",
    "m'acomiado", "m'estic acomiadant", "comiat",
    "aquesta és la meva comiat", "no tornaré",
    "ja no hi seré", "ja no existiré", "demà no hi seré",
    "aviat tot s'haurà acabat", "he regalat les meves coses",
    "ja m'he acomiadat dels meus amics",
}

# ---------------------------------------------------------------------------
# 8. ESCAPE / AVOIDANCE (ES + CA)
# ---------------------------------------------------------------------------
ESCAPE_AVOIDANCE = {
    # Spanish
    "quiero desaparecer", "quisiera desaparecer", "ojalá desapareciera",
    "quiero escapar", "escapar de todo", "huir de todo",
    "quiero que todo pare", "quiero que todo se acabe",
    "que todo termine", "dejar de sentir", "dejar de sufrir",
    "no quiero sentir nada", "no quiero sentir más",
    "dormirme para siempre", "no despertar", "no despertar más",
    "no despertar nunca más", "ojalá no me hubiera despertado",
    "desvanecerme", "esfumarme", "dejar de existir",
    "ya no quiero existir", "mejor si no existiera",
    # Catalan
    "vull desaparèixer", "voldria desaparèixer",
    "vull escapar", "escapar de tot", "fugir de tot",
    "vull que tot s'aturi", "vull que tot s'acabi",
    "que tot acabi", "deixar de sentir", "deixar de patir",
    "no vull sentir res", "adormir-me per sempre",
    "no despertar", "no despertar mai més",
    "esvair-me", "desaparèixer", "deixar d'existir",
    "ja no vull existir", "millor si no existís",
}

# ---------------------------------------------------------------------------
# 9. SELF-HARM (ES + CA) — non-suicidal self-injury markers
# ---------------------------------------------------------------------------
SELF_HARM = {
    # Spanish
    "cortarme", "cortarse", "me corto", "me he cortado",
    "hacerme cortes", "autolesión", "autolesionarme", "me lastimo",
    "me hago daño", "daño físico", "marcas en el cuerpo",
    "cicatrices", "quemarme", "golpearme", "castigarme",
    "me estoy castigando", "autocastigo", "lo merezco",
    # Catalan
    "tallar-me", "tallar-se", "em tallo", "m'he tallat",
    "fer-me talls", "autolesió", "autolesionar-me", "em faig mal",
    "mal físic", "marques al cos", "cicatrius",
    "cremar-me", "colpejar-me", "castigar-me",
    "m'estic castigant", "me l'ho mereixo",
}

# ---------------------------------------------------------------------------
# 10. PROTECTIVE FACTORS (ES + CA) — INVERSE signal
# ---------------------------------------------------------------------------
PROTECTIVE_FACTORS = {
    # Spanish
    "quiero mejorar", "quiero vivir", "ganas de vivir",
    "razones para vivir", "hay esperanza", "tengo esperanza",
    "saldrá bien", "va a mejorar", "estoy buscando ayuda",
    "pedir ayuda", "buscar ayuda", "llamar al médico",
    "ir al psicólogo", "ir al psiquiatra", "tengo apoyo",
    "me apoyan", "tengo familia", "tengo amigos",
    "no estoy solo", "no estoy sola", "cuido de mí",
    "me cuido", "seguir adelante", "fuerza", "resistir",
    "aguantar", "superarlo", "me va a pasar", "pasará",
    "mañana será mejor", "hay motivos para seguir",
    "tengo ganas", "ilusión", "proyecto de vida",
    # Catalan
    "vull millorar", "vull viure", "ganes de viure",
    "raons per viure", "hi ha esperança", "tinc esperança",
    "sortirà bé", "millorarà", "estic buscant ajuda",
    "demanar ajuda", "buscar ajuda", "anar al metge",
    "anar al psicòleg", "anar al psiquiatre", "tinc suport",
    "em donen suport", "tinc família", "tinc amics",
    "no estic sol", "no estic sola", "em cuido",
    "seguir endavant", "força", "resistir",
    "superar-ho", "em passarà", "passarà",
    "demà serà millor", "hi ha motius per continuar",
    "tinc ganes", "il·lusió", "projecte de vida",
}

# ---------------------------------------------------------------------------
# Aggregated dict for iteration
# ---------------------------------------------------------------------------
ALL_CATEGORIES: dict[str, set[str]] = {
    "suicidal_ideation":  SUICIDAL_IDEATION,
    "hopelessness":       HOPELESSNESS,
    "defeat_entrapment":  DEFEAT_ENTRAPMENT,
    "social_isolation":   SOCIAL_ISOLATION,
    "emotional_pain":     EMOTIONAL_PAIN,
    "negation_amplifiers":NEGATION_AMPLIFIERS,
    "temporal_finality":  TEMPORAL_FINALITY,
    "escape_avoidance":   ESCAPE_AVOIDANCE,
    "self_harm":          SELF_HARM,
    "protective_factors": PROTECTIVE_FACTORS,
}

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------
import re


def _tokenize(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^\w\sáéíóúüñàèìòùïçœ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lexicon_features(text: str, normalize: bool = True) -> dict[str, float]:
    """
    Return one score per lexicon category.

    Parameters
    ----------
    text : raw message text (body_anonimizado)
    normalize : divide hit count by word count (relative frequency)

    Returns
    -------
    dict with keys like 'lex_suicidal_ideation', 'lex_hopelessness', …
    protective_factors is returned as-is (caller may invert).
    """
    clean = _tokenize(text)
    words = clean.split()
    n_words = max(len(words), 1)
    features: dict[str, float] = {}

    for cat, terms in ALL_CATEGORIES.items():
        count = 0
        for term in terms:
            # Multi-word phrases: substring search in clean text
            if " " in term:
                count += clean.count(term)
            else:
                count += words.count(term)
        score = count / n_words if normalize else float(count)
        features[f"lex_{cat}"] = score

    return features


def lexicon_feature_matrix(texts: "list[str]", normalize: bool = True):
    """
    Apply lexicon_features to a list of texts.

    Returns
    -------
    pandas DataFrame  (n_texts × n_categories)
    """
    import pandas as pd
    rows = [lexicon_features(t, normalize=normalize) for t in texts]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick sanity check
    samples = [
        "No quiero seguir viviendo, ya no tiene sentido nada.",
        "Hola, estoy un poco triste hoy pero mañana será mejor.",
        "No puc més, vull desaparèixer per sempre.",
    ]
    df = lexicon_feature_matrix(samples)
    print(df.to_string())
    total = sum(len(v) for v in ALL_CATEGORIES.values())
    print(f"\nTotal lexicon entries: {total}")
    for cat, terms in ALL_CATEGORIES.items():
        print(f"  {cat}: {len(terms)} terms")
