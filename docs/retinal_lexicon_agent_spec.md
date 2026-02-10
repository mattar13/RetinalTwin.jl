# Retinal Digital Twin: Disease Lexicon & Diagnostic Agent Architecture

**Authors:** Matt Tarchick, Claude  
**Date:** 2026-02-08  
**Version:** 0.1.0  
**Companion files:** `retinal_disease_lexicon.json` (structured DB), `retinal_digital_twin_spec.md` (model spec), `retinal_twin_fitting_spec.md` (fitting spec)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Disease Lexicon — Human-Readable Reference](#2-disease-lexicon)
3. [Diagnostic Agent Architecture](#3-diagnostic-agent-architecture)
4. [Knowledge Graph Schema](#4-knowledge-graph-schema)
5. [Agent Reasoning Pipeline](#5-agent-reasoning-pipeline)
6. [Training & Retrieval Strategy](#6-training--retrieval-strategy)
7. [Query Interface Design](#7-query-interface-design)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Overview

This document specifies two interleaved systems:

**The Lexicon** is a structured knowledge base mapping retinal diseases to their cellular targets, ERG signatures, computational model parameter perturbations, disease staging, and available treatments. It serves as the ground truth for both the computational model's diagnostic inference and the LLM agent's reasoning.

**The Diagnostic Agent** is an LLM-powered reasoning layer that sits on top of the retinal digital twin. It ingests three types of evidence — (1) optimized model parameters from ERG fitting, (2) lexical patient data (questionnaires, clinical notes, fundus images), and (3) the disease lexicon — to produce clinically grounded diagnostic hypotheses, progression assessments, and treatment recommendations.

### The Full Stack

```
┌─────────────────────────────────────────────────────────┐
│                    QUERY INTERFACE                        │
│   "What diseases are consistent with this ERG?"          │
│   "How far has this patient's RP progressed?"            │
│   "What treatments are available for this genotype?"     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              DIAGNOSTIC AGENT (LLM)                      │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Parameter    │  │   Lexical    │  │   Lexicon    │  │
│  │  Interpreter  │  │  Integrator  │  │   Retrieval  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │           │
│  ┌──────▼─────────────────▼──────────────────▼───────┐  │
│  │           REASONING ENGINE                         │  │
│  │  • Differential diagnosis generation               │  │
│  │  • Disease staging / progression estimation        │  │
│  │  • Treatment matching (genotype → therapy)         │  │
│  │  • Uncertainty quantification                      │  │
│  └──────────────────────┬────────────────────────────┘  │
└─────────────────────────┼───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│              COMPUTATIONAL MODEL LAYER                    │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  ERG Fitting  │  │   Bayesian   │  │   Forward    │  │
│  │  Pipeline     │  │   Posterior   │  │   Simulator  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  193-ODE retinal column model                            │
│  48 biophysical parameters                               │
│  Staged fitting → MAP + NUTS posterior                    │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                  DATA SOURCES                             │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  ERG Traces   │  │  Patient     │  │  Fundus /    │  │
│  │  (ISCEV std)  │  │  History &   │  │  OCT Images  │  │
│  │              │  │  Questionnaire│  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Disease Lexicon — Human-Readable Reference

The following is the human-readable rendering of `retinal_disease_lexicon.json`. Each entry follows a consistent structure: disease identity → cellular pathology → ERG signature → model parameter mapping → staging → treatments.

---

### 2.1 Retinitis Pigmentosa (Rod-Cone Dystrophy)

**Category:** Inherited retinal dystrophy  
**Inheritance:** AD, AR, X-linked  
**Prevalence:** 1:4,000  
**Genes:** RHO, USH2A, RPGR, RP1, PRPF31, PRPH2, RPE65, PDE6A, PDE6B, NR2E3, CRB1, TULP1 (>70 genes total)  
**Progression:** Progressive rod-cone degeneration

#### Cellular Pathology

| Cell Type | Effect | Mechanism | Timing |
|-----------|--------|-----------|--------|
| **Rod** | Primary degeneration | Outer segment shortening → apoptosis; centripetal from periphery | Early |
| **Cone** | Secondary degeneration | Loss of RdCVF; oxidative stress from excess O₂ in thinned outer retina | Mid-Late |
| **Müller glia** | Reactive gliosis | GFAP upregulation, hypertrophy, glial seal, altered K⁺ buffering | Mid |
| **ON-bipolar** | Dendritic retraction | Inner retinal remodeling from loss of presynaptic input | Late |
| **RPE** | Secondary dysfunction | Pigment migration (bone spicules), RPE atrophy | Mid-Late |

#### ERG Signature

**Diagnostic pattern:** Scotopic responses reduced/absent with **delayed implicit times** before photopic changes. b-wave implicit time delay is the hallmark distinguishing progressive RP from CSNB. OPs are lost early.

| Component | Amplitude | Implicit Time | Notes |
|-----------|-----------|---------------|-------|
| Scotopic a-wave | ↓↓↓ to absent | Delayed | Earliest and most severely affected |
| Scotopic b-wave | ↓↓↓ to absent | **Delayed** | Implicit time delay is cardinal feature |
| Scotopic OPs | Absent early | — | Lost before a/b become unrecordable |
| Photopic b-wave | ↓ to ↓↓ | Delayed | Affected later than scotopic |
| 30 Hz flicker | ↓↓ | **Delayed** | Sensitive progression marker |

**Progression monitoring:** 30 Hz flicker implicit time + amplitude decline rate; correlates with EZ band area on OCT.

#### Model Parameter Mapping

| Parameter | Direction | Magnitude | Stage | Rationale |
|-----------|-----------|-----------|-------|-----------|
| `N_rod` | ↓ | 0.1–1.0 (fraction remaining) | Early→Late | Progressive rod loss |
| `g_CNG_rod` | ↓ | 0.3–0.8× | Early | Outer segment shortening |
| `τ_R_rod` | ↑ | 1.5–3× | Early | Impaired phototransduction kinetics |
| `N_cone` | ↓ | 0.3–1.0 | Mid→Late | Secondary cone loss |
| `g_Kir_end` | ↓ | 0.5–0.8× | Mid | Müller gliosis alters K⁺ buffering |
| `w_rod` | ↓ | ∝ N_rod | All | Weight tracks rod population |
| `w_on` | ↓ | 0.5–0.8× | Late | Inner retinal remodeling |

#### Disease Staging

| Stage | Clinical | ERG | Model State |
|-------|----------|-----|-------------|
| **Early** | Night blindness, mid-peripheral VF loss | Scotopic b-wave ↓ and delayed; photopic near-normal | N_rod ~0.5–0.7, g_CNG ~0.6× |
| **Moderate** | Ring scotoma, bone spicules, attenuated vessels | Scotopic absent/near-absent; photopic ↓ and delayed | N_rod ~0.1–0.3, N_cone ~0.5–0.7 |
| **Advanced** | Tunnel vision, macular involvement | All responses extinguished | N_rod ≈ 0, N_cone ~0.1–0.3 |

#### Treatments

| Treatment | Type | Status | Target | Key Detail |
|-----------|------|--------|--------|------------|
| **Luxturna** (voretigene neparvovec) | Gene therapy | FDA approved 2017 | RPE65 | AAV2 subretinal injection; restores visual cycle |
| RPGR gene therapy | Gene therapy | Phase 2/3 | RPGR (X-linked RP) | Multiple trials ongoing |
| OCU400 | Gene-agnostic modifier | Phase 1/2 | NR2E3 | Works across multiple RP genotypes |
| Vitamin A palmitate | Nutritional | Standard of care | — | ~20% slower ERG decline/year |
| Optogenetic therapy | Optogenetic | Phase 1/2 | Genotype-independent | For advanced disease; bypasses photoreceptors |

---

### 2.2 Congenital Stationary Night Blindness — Complete (Type 1)

**Category:** Inherited retinal dystrophy (stationary)  
**Inheritance:** X-linked, AR  
**Genes:** NYX, GRM6, TRPM1, GPR179, LRIT3  
**Progression:** **Stationary** — does not worsen over time

#### Cellular Pathology

| Cell Type | Effect | Mechanism |
|-----------|--------|-----------|
| **ON-Bipolar** | Signaling defect | mGluR6 → TRPM1 cascade completely non-functional; **photoreceptors are structurally normal** |

#### ERG Signature

**Diagnostic pattern:** 🔴 **ELECTRONEGATIVE ERG** — preserved a-wave with severely attenuated/absent b-wave. **Normal implicit times** distinguish from RP.

| Component | Amplitude | Implicit Time |
|-----------|-----------|---------------|
| Scotopic a-wave | **Normal** | Normal |
| Scotopic b-wave | **Absent/severely reduced** | Normal when present |
| OPs | Absent | — |
| 30 Hz flicker | Reduced | **Normal** (vs. delayed in RP) |

#### Model Parameter Mapping

| Parameter | Change | Rationale |
|-----------|--------|-----------|
| `g_TRPM1` | → 0 | TRPM1 channel completely non-functional |
| `τ_mGluR6` | → ∞ | No signal transduction |
| `w_on` | → ~0 | ON-pathway contributes nothing |
| All photoreceptor params | **Normal** | Photoreceptors intact |

---

### 2.3 Congenital Stationary Night Blindness — Incomplete (Type 2)

**Genes:** CACNA1F, CABP4, CACNA2D4  
**Key distinction from Complete CSNB:** Residual b-wave present; defect is at the **photoreceptor synaptic Ca²⁺ channel**, not the ON-bipolar signal cascade.

| Component | Complete CSNB | Incomplete CSNB |
|-----------|---------------|-----------------|
| a-wave | Normal | Normal to mildly reduced |
| b-wave | Absent | Reduced (residual present) |
| 30 Hz flicker | Reduced | Reduced with bifurcated trough |
| OPs | Absent | Reduced |

**Model:** `τ_Glu_on` ↑ 3–5× (impaired Ca-dependent glutamate release), `g_TRPM1` ↓ to 0.3–0.5× (reduced but not absent).

---

### 2.4 Leber Congenital Amaurosis (LCA)

**Category:** Early-onset severe retinal dystrophy  
**Prevalence:** 1:30,000–1:80,000  
**Onset:** Birth to first year  
**Genes:** RPE65, CEP290, GUCY2D, CRX, CRB1, AIPL1, RPGRIP1, LCA5, NMNAT1, RDH12 (25 genes total)

#### ERG Signature

**Diagnostic pattern:** Severely abnormal or **extinguished ERG from birth**. This is the defining feature. Some genotypes (CEP290) retain residual cone responses.

All scotopic and photopic responses: **absent or profoundly reduced**.

#### Model Parameter Mapping

Gene-dependent — examples:

- **RPE65:** `η_rod` → 0 (no 11-cis retinal → zero quantum efficiency)
- **GUCY2D:** `α_G` → 0 (no cGMP synthesis in phototransduction)
- **AIPL1:** `N_rod/N_cone` → near-zero (photoreceptor degeneration from birth)

#### Treatments

| Treatment | Status | Target Gene |
|-----------|--------|-------------|
| **Luxturna** | FDA approved 2017 | RPE65 (LCA type 2) |
| EDIT-101 (CRISPR) | Phase 1/2 | CEP290 (LCA type 10) |
| OPGx-LCA5 | Phase 1/2 | LCA5 (LCA type 5) |
| AIPL1 gene therapy | Phase 1 (2025) | AIPL1 |

---

### 2.5 Stargardt Disease

**Category:** Inherited macular dystrophy  
**Prevalence:** 1:8,000–1:10,000  
**Gene:** ABCA4 (AR); ELOVL4, PROM1 (AD rare)  
**Mechanism:** ABCA4 dysfunction → A2E bisretinoid accumulates in RPE → RPE death → secondary photoreceptor loss

#### ERG Signature (Lois Classification)

| Group | Full-field ERG | PERG | mfERG | Clinical |
|-------|---------------|------|-------|----------|
| **1** | **Normal** | Reduced P50 | Central depression | Macular only |
| **2** | Photopic ↓ | Reduced | Broad depression | Generalized cone + macular |
| **3** | Scotopic AND photopic ↓ | Reduced | Broad | Generalized rod-cone |

**Key:** Full-field ERG can be completely normal in early Stargardt. **PERG** is the most sensitive ERG measure. **EOG** is abnormal (suppressed Arden ratio).

#### Model: `τ_RPE` ↑ 2–5×, `α_K_RPE` ↓ 0.3–0.7×, `N_cone` ↓ centrally (mid stage), `N_rod` ↓ (Group 3 only, late)

#### Treatment pipeline

SB-007 (dual-vector ABCA4, IND approved Dec 2024); visual cycle modulators (emixustat, ALK-001); complement inhibition.

---

### 2.6 Best Vitelliform Macular Dystrophy

**Gene:** BEST1 (bestrophin-1, Ca²⁺-activated Cl⁻ channel in RPE)  
**Inheritance:** AD

#### ERG Signature

**Pathognomonic:** **Normal full-field ERG** + **abnormal EOG** (Arden ratio < 1.5). This combination is virtually diagnostic.

**Model note:** Current model uses K⁺-based RPE representation. Would need Cl⁻ conductance extension for bestrophin-1. Current proxy: `τ_RPE` ↑ 2–4×, c-wave amplitude ↓.

---

### 2.7 X-Linked Juvenile Retinoschisis

**Gene:** RS1 (retinoschisin, cell adhesion protein)  
**Prevalence:** 1:5,000–1:25,000 males

#### ERG Signature

🔴 **ELECTRONEGATIVE ERG** — same pattern as complete CSNB. Key differential: **foveal schisis on OCT**, spoke-wheel macular pattern, splitting at INL level.

**Model:** `w_on` ↓ 0.2–0.5× (disrupted bipolar signaling through schisis), `w_muller` ↓ 0.3–0.6× (Müller K⁺ buffering disrupted).

**Treatment:** RS1 gene therapy (intravitreal AAV), Phase 1/2.

---

### 2.8 Diabetic Retinopathy

**Category:** Vascular  
**Prevalence:** ~30% of all diabetics

#### Cellular Pathology — Critical Insight

Diabetic retinopathy is **not just a vascular disease**. Neural retinal dysfunction (particularly amacrine cells and RGCs) is detectable by ERG **before any visible vascular lesions appear**. This makes ERG + digital twin a powerful preclinical biomarker.

| Cell Type | Effect | Timing |
|-----------|--------|--------|
| **AII Amacrine** | Dysfunction | Preclinical — **earliest neural change** |
| **GABAergic Amacrine** | Dysfunction/loss | Early |
| **RGC** | Apoptosis | Early (GCL thinning on OCT) |
| **Müller glia** | Gliosis, VEGF release | Early-Mid |
| **Rod** | Reduced sensitivity | Preclinical |
| **Vasculature** | Microangiopathy → neovascularization | Progressive |

#### ERG Signature

**Key finding:** **Oscillatory potential reduction and delay is the earliest and most sensitive ERG indicator of diabetic retinopathy**, detectable before clinical fundus changes.

| Component | Change | Clinical Significance |
|-----------|--------|----------------------|
| **OPs** | ↓↓ amplitude, ↑ implicit time | **Most sensitive** — precedes visible DR |
| Scotopic a-wave | Mildly ↓ and delayed | Rod sensitivity reduced early |
| Scotopic b-wave | ↓, delayed | Later than OPs |
| **PhNR** | ↓ | RGC dysfunction marker |
| 30 Hz flicker | ↓, delayed | Predicts progression to proliferative DR |

#### Model Parameter Mapping

| Parameter | Direction | Magnitude | Stage |
|-----------|-----------|-----------|-------|
| `g_Ca_a2` | ↓ | 0.5–0.8× | Early (→ OP reduction) |
| `g_Ca_gaba` | ↓ | 0.5–0.8× | Early |
| `τ_Gly` | ↑ | 1.5–2× | Early (→ OP delay) |
| `τ_GABA` | ↑ | 1.5–2× | Early |
| `g_Kir_end` | ↓ | 0.6–0.8× | Mid (Müller gliosis) |
| `g_CNG_rod` | ↓ | 0.8–0.95× | Early (subtle) |

#### Staging with ERG Correlation

| Stage | Fundus | ERG | Model |
|-------|--------|-----|-------|
| **Preclinical** | Normal | OPs ↓, subtle rod changes | Amacrine params shifted |
| Mild NPDR | Microaneurysms | OPs further ↓, b-wave beginning to reduce | + Müller shifts |
| Mod-Severe NPDR | Hemorrhages, CWS, venous beading | All scotopic ↓, 30 Hz delayed | Broad inner retinal shifts |
| Proliferative | Neovascularization | Severely ↓ | Severe inner retinal dysfunction |

---

### 2.9 Central Retinal Vein Occlusion (CRVO)

**ERG diagnostic role:** b:a ratio on bright flash scotopic ERG **predicts ischemic vs. non-ischemic CRVO**. b:a < 1 (electronegative) → ischemic type, high risk of neovascularization. This is one of the most clinically actionable ERG findings.

**Model:** `w_on` ↓ 0.3–0.8× (inner retinal ischemia), `g_Ca_a2` ↓ 0.3–0.7× (amacrine ischemia → OP loss).

---

### 2.10 Age-Related Macular Degeneration — Dry

**Key ERG insight:** Full-field ERG is often **normal** in early-moderate AMD because the macular lesion is too small relative to total retina. **Pattern ERG** and **multifocal ERG** are the relevant modalities. mfERG implicit time delays in perilesional zones predict geographic atrophy expansion.

**Model:** `τ_RPE` ↑ 2–5× (RPE dysfunction), `N_cone` ↓ centrally (mid-late), `η_rod` ↓ 0.7–0.9× (dark adaptation impairment — earliest functional sign).

**Treatments:** Pegcetacoplan (Syfovre, FDA 2023) and avacincaptad pegol (Izervay, FDA 2023) for geographic atrophy; AREDS2 supplementation.

---

### 2.11 Enhanced S-Cone Syndrome

**Gene:** NR2E3  
**ERG:** 🟢 **PATHOGNOMONIC** — Supranormal S-cone ERG with absent rod response. One of the few diseases diagnosable solely from ERG pattern.  
**Model note:** Current model lacks S-cone vs. L/M cone subtype distinction. Would need cone subtype extension.

---

### 2.12 Achromatopsia (Rod Monochromacy)

**Genes:** CNGA3, CNGB3, GNAT2, PDE6C  
**ERG:** Normal scotopic responses; **absent photopic and 30 Hz flicker**. Clean dichotomy.  
**Model:** `g_CNG_cone` → 0 (for CNGA3/CNGB3); `γ_cone` → 0 (for GNAT2/PDE6C).  
**Treatment:** CNGA3 and CNGB3 gene therapy both in Phase 1/2.

---

### 2.13 Retinopathy of Prematurity (ROP)

**ERG role:** Even in premature infants without ROP, rod ERG shows maturational delay. In treated/regressed ROP, **long-term ERG deficits persist** (reduced scotopic, reduced OPs), suggesting lasting inner retinal dysfunction.

**Model:** `N_rod` mildly ↓ (0.7–0.9, immature), `g_Ca_a2` ↓ 0.6–0.8× (inner retinal immaturity/ischemia).

---

### 2.14 Differential Diagnosis Decision Trees

The agent uses these patterns to narrow differential diagnoses from ERG signatures:

#### Electronegative ERG (a-wave > b-wave)

```
                    Electronegative ERG
                          │
              ┌───────────┼───────────┐
              │           │           │
         Congenital?   Acute onset?  Male + 
              │           │       foveal schisis?
              │           │           │
        ┌─────┴─────┐    │       XLRS (RS1)
        │           │    │
   Complete    Incomplete │
   CSNB         CSNB     │
  (NYX/GRM6)  (CACNA1F)  │
                     ┌────┴────┐
                     │         │
                  Ischemic   Melanoma
                   CRVO     Assoc. Retinopathy
```

#### Absent/Extinguished ERG

```
                  Extinguished ERG
                        │
              ┌─────────┼─────────┐
              │         │         │
          Infantile  Progressive  Acute adult
           onset      history      onset
              │         │           │
            LCA     Advanced    Cancer-Assoc.
                      RP        Retinopathy
```

#### OPs Selectively Reduced

```
             OPs selectively reduced
             (a-wave and b-wave preserved)
                        │
              ┌─────────┼─────────┐
              │         │         │
           Diabetic?  Acute?    Sickle cell?
              │         │           │
             DR       CRVO/     SC retinopathy
          (even w/o   BRVO
          visible DR)
```

---

## 3. Diagnostic Agent Architecture

### 3.1 Design Philosophy

The agent operates as a **clinical reasoning assistant**, not a replacement for clinical judgment. It synthesizes evidence from three independent channels that each constrain the diagnostic space differently:

1. **Computational channel** — Biophysical model parameters estimated from ERG fitting
2. **Lexical channel** — Patient history, questionnaires, clinical notes, genetic testing
3. **Imaging channel** — Fundus photography, OCT, autofluorescence (future: multimodal analysis)

The key insight is that **these channels disambiguate each other**. Multiple diseases produce similar ERG parameter shifts (e.g., electronegative ERGs), but the lexical/imaging context narrows the differential dramatically.

### 3.2 Agent Components

```julia
# Core agent structure
struct DiagnosticAgent
    # Knowledge stores
    lexicon::DiseaseLexicon              # Structured disease knowledge base
    literature_index::RAGIndex           # Embedded PubMed/literature corpus
    treatment_registry::TreatmentDB     # Current clinical trials + approved therapies
    
    # Model interface
    digital_twin::RetinalTwin           # The 193-ODE model
    fitting_result::FittingResult       # Optimized params + posterior
    
    # Patient data
    patient::PatientRecord              # Demographics, history, questionnaire
    erg_data::ERGDataSet                # Raw ERG traces
    imaging::ImagingData                # Fundus, OCT, AF (optional)
    
    # LLM backbone
    llm::LLMInterface                   # Claude API or local model
    prompt_templates::PromptLibrary     # Structured prompts for each reasoning task
end
```

### 3.3 Three-Channel Evidence Architecture

#### Channel 1: Parameter Interpreter

Takes the fitted parameter set θ* (and posterior distributions from Bayesian inference) and translates them into clinically interpretable statements.

```julia
struct ParameterInterpretation
    # Which parameters deviate significantly from healthy baseline?
    anomalous_params::Vector{ParamAnomaly}
    
    # What cell types are implicated?
    affected_cells::Vector{CellImplication}
    
    # What disease patterns match?
    candidate_diseases::Vector{DiseaseMatch}
    
    # Confidence from posterior width
    parameter_confidence::Dict{Symbol, Float64}
end

struct ParamAnomaly
    param::Symbol           # e.g., :g_TRPM1
    fitted_value::Float64   # MAP estimate
    healthy_range::Tuple{Float64, Float64}  # Population normal range
    posterior_ci::Tuple{Float64, Float64}   # 95% credible interval
    z_score::Float64        # Deviation from healthy mean in SDs
    direction::Symbol       # :elevated, :reduced, :normal
end

struct CellImplication
    cell_type::Symbol       # e.g., :on_bc
    evidence_params::Vector{Symbol}  # Which params implicate this cell
    dysfunction_type::Symbol  # :reduced_function, :absent, :hyperactive
    confidence::Float64      # 0-1
end

struct DiseaseMatch
    disease_id::String       # Lexicon disease ID
    match_score::Float64     # 0-1 based on parameter pattern overlap
    matching_params::Vector{Symbol}
    conflicting_params::Vector{Symbol}
    stage_estimate::String   # Best-fit disease stage
end
```

**How it works:**

1. Compare each fitted parameter to a population-level healthy baseline (from literature/normative data)
2. Identify significant deviations (z > 2 or outside 95% CI of healthy population)
3. Map deviations to cell types using `cell_types` in the lexicon
4. Pattern-match against all `model_parameter_changes` entries in the disease database
5. Score each disease by cosine similarity between observed parameter deviation vector and expected deviation vector for that disease/stage
6. Rank candidate diseases

#### Channel 2: Lexical Integrator

Processes unstructured patient data and extracts clinically relevant features for diagnosis.

```julia
struct PatientRecord
    # Demographics
    age::Int
    sex::Symbol
    ethnicity::String
    
    # Clinical history
    chief_complaint::String           # e.g., "night blindness since childhood"
    symptom_onset::String             # "congenital", "age 10", "6 months ago"
    symptom_progression::Symbol       # :stationary, :progressive, :acute
    family_history::Vector{String}    # Pedigree information
    
    # Questionnaire responses
    night_vision::Symbol              # :normal, :mild_difficulty, :severe
    peripheral_vision::Symbol         # :normal, :constricted, :tunnel
    color_vision::Symbol              # :normal, :reduced, :absent
    photophobia::Symbol               # :none, :mild, :severe
    nystagmus::Bool
    
    # Medical history
    systemic_conditions::Vector{String}  # "type 2 diabetes", "hypertension"
    medications::Vector{String}
    prior_ocular_surgery::Vector{String}
    
    # Genetic testing
    genetic_test_result::Union{Nothing, GeneticResult}
    
    # Visual acuity
    bcva_od::String  # e.g., "20/200"
    bcva_os::String
end

struct GeneticResult
    gene::String
    variant::String
    zygosity::Symbol          # :homozygous, :heterozygous, :compound_het
    pathogenicity::Symbol     # :pathogenic, :likely_pathogenic, :VUS
    inheritance::Symbol
end
```

**LLM extraction pipeline:**

The LLM agent processes clinical notes and questionnaire responses through structured extraction prompts:

```
SYSTEM: You are a retinal disease diagnostic assistant. Extract structured 
clinical features from the following patient data. For each feature, assess
relevance to retinal disease diagnosis.

PATIENT DATA: {raw_clinical_notes}

Extract:
1. Symptom onset and timeline
2. Progressive vs. stationary indicators  
3. Rod vs. cone involvement pattern
4. Systemic disease associations
5. Family history pattern (inheritance mode)
6. Red flags for specific diagnoses
```

#### Channel 3: Lexicon Retrieval (RAG)

The lexicon serves as a curated, authoritative knowledge base that the LLM accesses via retrieval-augmented generation. Two retrieval modes:

**Mode A: Structured query** — Direct lookup against JSON schema:
```julia
function query_lexicon(lexicon::DiseaseLexicon; 
                       cell_type=nothing, 
                       erg_pattern=nothing,
                       gene=nothing,
                       symptom=nothing)
    # Filter diseases by matching criteria
    # Returns ranked list of matching disease entries
end
```

**Mode B: Semantic retrieval** — For complex or ambiguous queries, embed the question and retrieve from the literature index:
```julia
function semantic_search(index::RAGIndex, query::String; top_k=10)
    # Embed query using sentence transformer
    # Retrieve top-k chunks from indexed literature
    # Re-rank with cross-encoder
    # Return relevant passages with citations
end
```

### 3.4 Evidence Fusion & Reasoning

The three channels produce independent evidence streams. The reasoning engine fuses them:

```julia
struct DiagnosticReport
    # Primary output: ranked differential diagnosis
    differential::Vector{DiagnosisHypothesis}
    
    # Progression assessment (for identified disease)
    staging::Union{Nothing, StageAssessment}
    
    # Treatment recommendations
    treatments::Vector{TreatmentRecommendation}
    
    # Uncertainty assessment
    overall_confidence::Symbol  # :high, :moderate, :low
    conflicting_evidence::Vector{String}
    recommended_additional_tests::Vector{String}
    
    # Reasoning trace
    reasoning_chain::String  # Human-readable explanation of logic
end

struct DiagnosisHypothesis
    disease_id::String
    disease_name::String
    probability::Float64            # Composite confidence
    
    # Evidence breakdown
    parameter_evidence::Float64     # From channel 1
    lexical_evidence::Float64       # From channel 2
    imaging_evidence::Float64       # From channel 3
    
    supporting_evidence::Vector{String}
    conflicting_evidence::Vector{String}
    
    # If genetic testing available
    genotype_match::Union{Nothing, Bool}
end
```

**Fusion strategy:**

The agent doesn't just average confidence scores. It uses the LLM's reasoning capacity to perform **clinical-style differential diagnosis**:

```
SYSTEM: You are an expert retinal electrophysiologist and clinical geneticist. 
You have three sources of evidence about this patient. Reason through the 
differential diagnosis step by step.

PARAMETER EVIDENCE:
{parameter_interpretation — which params are abnormal, what cells are implicated}

LEXICAL EVIDENCE:
{patient history, onset, symptoms, family history, genetic testing if available}

IMAGING EVIDENCE:
{fundus findings, OCT features — if available}

LEXICON CONTEXT:
{top 5 disease entries from lexicon retrieval, including ERG signatures and 
parameter mappings}

DIFFERENTIAL DIAGNOSIS RULES:
{relevant decision trees from the lexicon}

Instructions:
1. List all diseases consistent with the parameter evidence alone
2. Use lexical evidence to narrow: which diseases are excluded by onset age, 
   inheritance pattern, progression status, or systemic associations?
3. Use imaging evidence to further narrow (if available)
4. For remaining candidates, compare expected vs. observed parameter patterns 
   in detail
5. Assign probability estimates to each remaining hypothesis
6. Identify which additional test would best disambiguate remaining candidates
7. If a disease is identified with high confidence, estimate the disease stage
8. Match the identified disease + genotype (if known) to available treatments
```

---

## 4. Knowledge Graph Schema

The lexicon is structured as a knowledge graph with the following entity-relationship model:

```
┌──────────┐     affects      ┌──────────┐    generates    ┌──────────┐
│ Disease  │────────────────▶│ Cell Type │───────────────▶│   ERG    │
│          │  (mechanism,    │          │  (contributes   │Component │
│          │   timing,       │          │   to)           │          │
│          │   severity)     │          │                 │          │
└──────────┘                 └──────────┘                 └──────────┘
     │                            │                            │
     │ perturbs                   │ has_params                 │ measured_as
     ▼                            ▼                            ▼
┌──────────┐                ┌──────────┐                ┌──────────┐
│  Model   │                │ Parameter│                │  Feature │
│Parameter │                │  Group   │                │(amp, IT, │
│ Change   │                │          │                │ freq)    │
└──────────┘                └──────────┘                └──────────┘
     │
     │ reversed_by
     ▼
┌──────────┐
│Treatment │
│(gene Rx, │
│ pharma)  │
└──────────┘
```

### Bidirectional Mapping

The critical design feature is **bidirectionality**:

**Forward (Disease → Parameters):** Given a disease diagnosis, what parameter changes should the model exhibit?  
**Inverse (Parameters → Disease):** Given observed parameter anomalies, what diseases are consistent?

The inverse direction is the harder problem because it's many-to-one. This is where the agent's reasoning capacity is essential — it uses the lexical context to disambiguate.

### Parameter Deviation Fingerprints

Each disease has a characteristic "fingerprint" in parameter space:

```julia
struct DiseaseFingerprint
    disease_id::String
    stage::String
    
    # Expected parameter deviations from healthy baseline
    # Stored as z-scores (SDs from healthy mean)
    deviations::Dict{Symbol, NormalizedDeviation}
end

struct NormalizedDeviation
    expected_z::Float64       # Expected z-score for this disease/stage
    tolerance::Float64        # How much variation is expected
    required::Bool            # Must this param be abnormal for diagnosis?
end
```

The matching algorithm computes a weighted cosine similarity between the observed deviation vector and each disease fingerprint, with required deviations acting as hard constraints.

---

## 5. Agent Reasoning Pipeline

### 5.1 Full Pipeline Flow

```
Patient presents with ERG data + clinical information
                    │
                    ▼
    ┌───────────────────────────────┐
    │  1. ERG FITTING               │
    │  • Load ERG traces            │
    │  • Run staged fitting pipeline│
    │  • Get MAP estimate + posterior│
    │  • Extract parameter set θ*   │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  2. PARAMETER INTERPRETATION  │
    │  • Compare θ* to healthy      │
    │    baseline                   │
    │  • Identify anomalous params  │
    │  • Map to cell types          │
    │  • Pattern-match to diseases  │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌─────────┐  ┌──────────┐  ┌──────────┐
│Parameter│  │ Lexical  │  │ Lexicon  │
│Evidence │  │ Evidence │  │ Retrieval│
│(θ* → Dx)│  │(Hx → Dx)│  │(DB → Dx) │
└────┬────┘  └────┬─────┘  └────┬─────┘
     │            │              │
     └────────────┼──────────────┘
                  │
                  ▼
    ┌───────────────────────────────┐
    │  3. EVIDENCE FUSION (LLM)     │
    │  • Differential diagnosis     │
    │  • Bayesian-style reasoning   │
    │  • Conflict resolution        │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  4. DIAGNOSTIC OUTPUT          │
    │  • Ranked differential        │
    │  • Stage assessment           │
    │  • Treatment matching         │
    │  • Recommended next tests     │
    │  • Reasoning trace            │
    └───────────────────────────────┘
```

### 5.2 Query Types

The agent supports three primary query types:

#### Query 1: "What diseases are present?"

Input: ERG data + patient history  
Process: Full pipeline (fit → interpret → fuse → diagnose)  
Output: Ranked differential with confidence scores and reasoning

#### Query 2: "How far has this disease progressed?"

Input: ERG data + known diagnosis  
Process: Fit model → compare parameter values to staging table for known disease → estimate stage  
Output: Stage assessment with quantitative parameter-based severity scores

**Key innovation:** The digital twin provides **continuous** staging, not just categorical. Instead of "early" vs. "moderate," it can report "N_rod = 0.42 (estimated 58% rod loss), consistent with early-to-moderate RP."

The Bayesian posterior provides uncertainty: "N_rod 95% CI [0.35, 0.51], suggesting 49–65% rod loss."

#### Query 3: "What treatments are available?"

Input: Diagnosis + genotype (if known)  
Process: Look up disease in lexicon → filter treatments by eligibility criteria → retrieve current trial status from treatment registry → LLM synthesizes recommendation  
Output: Treatment options ranked by eligibility match, with trial enrollment information

### 5.3 Prompt Architecture

The agent uses structured prompts organized as a library:

```julia
struct PromptLibrary
    system_prompt::String                    # Core identity and constraints
    parameter_interpretation::String          # θ* → clinical meaning
    differential_diagnosis::String            # Evidence fusion → Dx
    staging_assessment::String               # Known Dx + θ* → stage
    treatment_matching::String               # Dx + genotype → Rx
    literature_synthesis::String             # RAG results → coherent answer
    uncertainty_communication::String         # Posterior → patient-friendly uncertainty
end
```

**System prompt core:**

```
You are a retinal electrophysiology diagnostic assistant embedded within the 
Retinal Digital Twin system. You have access to:

1. A biophysical model of the retina (193 ODEs, 48 parameters) that has been 
   fitted to the patient's ERG data
2. A curated disease lexicon mapping retinal diseases to cellular pathology, 
   ERG signatures, model parameter changes, and treatments
3. The patient's clinical history, questionnaire responses, and imaging data

Your role is to:
- Interpret model parameters in clinical terms
- Generate differential diagnoses by fusing computational and clinical evidence
- Estimate disease stage and progression
- Match patients to available treatments
- Communicate uncertainty honestly
- Provide citations to the lexicon and literature

You are NOT a replacement for clinical judgment. Always frame outputs as 
"evidence suggests" rather than definitive diagnoses. Flag cases where evidence 
is conflicting or insufficient.

CRITICAL: Never recommend treatments without noting that clinical decisions 
require physician oversight. Your role is to surface evidence and organize 
reasoning, not to prescribe.
```

---

## 6. Training & Retrieval Strategy

### 6.1 Knowledge Sources

| Source | Content | Format | Update Frequency |
|--------|---------|--------|-----------------|
| Disease Lexicon (this doc) | Curated disease-cell-ERG-param mappings | JSON + Markdown | Manual (quarterly) |
| PubMed literature | ERG findings in retinal disease | Indexed PDFs | Monthly scrape |
| ClinicalTrials.gov | Active gene therapy trials | Structured API | Weekly |
| ISCEV standards | ERG recording/interpretation guidelines | PDF | Annual |
| OMIM / RetNet | Gene-disease associations | Structured DB | Monthly |
| Normative ERG data | Healthy population parameter baselines | CSV/statistical models | As studies publish |

### 6.2 RAG Architecture

```
┌─────────────────────────────────────────┐
│          LITERATURE CORPUS               │
│                                          │
│  PubMed abstracts + full-text           │
│  ISCEV standards                        │
│  Review articles on ERG in disease      │
│  Gene therapy trial publications        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│          INDEXING PIPELINE               │
│                                          │
│  1. Chunk documents (512 token chunks)  │
│  2. Embed with bio/medical model        │
│     (PubMedBERT or similar)             │
│  3. Store in vector DB (FAISS/Qdrant)   │
│  4. Tag with metadata:                  │
│     - disease_ids mentioned             │
│     - cell_types mentioned              │
│     - erg_components mentioned          │
│     - publication_year                  │
│     - study_type (RCT, case, review)    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│          RETRIEVAL PIPELINE              │
│                                          │
│  Query: "What ERG changes are seen      │
│          in early diabetic retinopathy   │
│          before visible lesions?"        │
│                                          │
│  1. Embed query                         │
│  2. Hybrid search:                      │
│     - Vector similarity (semantic)      │
│     - Keyword filter (disease_id, etc.) │
│  3. Re-rank with cross-encoder          │
│  4. Inject top-k chunks into LLM prompt │
│  5. LLM synthesizes with citations      │
└─────────────────────────────────────────┘
```

### 6.3 Lexicon as Structured Knowledge Layer

The curated JSON lexicon sits above the RAG layer. It provides **authoritative, validated** mappings that the LLM should trust over noisy RAG retrieval results. The hierarchy:

1. **Lexicon (highest authority):** Curated disease-param mappings, ERG signatures, staging
2. **RAG retrieval (supporting evidence):** Literature context, detailed mechanisms, case reports
3. **LLM training knowledge (lowest priority):** General biomedical knowledge as fallback

The agent prompt explicitly instructs the LLM to defer to the lexicon for parameter mappings and ERG signatures, use RAG for mechanistic detail and edge cases, and rely on training knowledge only when lexicon and RAG are silent.

### 6.4 Fine-Tuning Considerations (Future)

For production deployment, the LLM backbone could be fine-tuned on:

- Clinical ERG interpretation reports (with physician annotations)
- Disease-ERG association datasets
- Parameter fitting results mapped to confirmed diagnoses
- Treatment matching examples

This would be a second phase after the lexicon + RAG system is validated.

---

## 7. Query Interface Design

### 7.1 Natural Language Interface

The system accepts natural language queries and routes them to the appropriate pipeline:

```julia
function process_query(agent::DiagnosticAgent, query::String)
    # Classify query type
    query_type = classify_query(agent.llm, query)
    
    if query_type == :diagnostic
        return run_diagnostic_pipeline(agent)
    elseif query_type == :staging
        return run_staging_assessment(agent)
    elseif query_type == :treatment
        return run_treatment_matching(agent)
    elseif query_type == :literature
        return run_literature_query(agent, query)
    elseif query_type == :parameter_explanation
        return explain_parameters(agent)
    end
end
```

### 7.2 Example Interactions

**Example 1: Diagnostic query**

```
User: "I have a 12-year-old male with night blindness since age 5. Family 
history shows affected maternal uncle. ERG shows electronegative scotopic 
response with normal a-wave implicit time. What could this be?"

Agent reasoning:
- Parameter evidence: g_TRPM1 near zero → ON-bipolar pathway absent
- Lexical evidence: male + maternal uncle = X-linked; congenital/early onset; 
  stationary complaint (no progression mentioned); normal a-wave timing
- Lexicon match: Electronegative ERG differential → CSNB complete (NYX), 
  XLRS (RS1), incomplete CSNB (CACNA1F)
- Narrowing: X-linked inheritance + normal implicit times + no schisis on OCT 
  (would need to check) → most likely CSNB type 1 (NYX)

Output: "The ERG pattern and clinical history are most consistent with 
Complete Congenital Stationary Night Blindness (Schubert-Bornschein type), 
likely X-linked. The electronegative ERG with normal a-wave and normal 
implicit times points to an ON-bipolar signaling defect rather than 
progressive photoreceptor disease. The X-linked family history suggests 
NYX or CACNA1F.

Recommended: 
- OCT to rule out X-linked retinoschisis (foveal schisis)
- Genetic testing for NYX (complete) vs. CACNA1F (incomplete)
- This is a stationary condition — visual function should remain stable."
```

**Example 2: Progression query**

```
User: "This patient has confirmed USH2A-related RP. How far has the 
disease progressed based on the fitted parameters?"

Agent reasoning:
- θ* shows: N_rod = 0.28, N_cone = 0.72, g_CNG_rod = 0.45, τ_R = 2.1×
- Posterior: N_rod 95% CI [0.21, 0.36]
- Lexicon staging: N_rod 0.1-0.3 maps to moderate stage
- Additional: N_cone still relatively preserved → mid-stage transition

Output: "Based on the fitted parameters, this patient's USH2A-RP is in 
the moderate stage:
- Estimated rod loss: 72% (N_rod = 0.28, 95% CI: 64-79%)
- Cone preservation: ~72% remaining (secondary loss has begun)
- Phototransduction kinetics: τ_R elevated to 2.1× normal

This is consistent with the transition from moderate to advanced disease. 
The cones are still relatively preserved, suggesting a window for 
potential therapeutic intervention. At the current estimated rate, 
significant cone involvement may progress over the next 5-10 years."
```

**Example 3: Treatment query**

```
User: "What treatments are available for this patient's genotype?"

Agent reasoning:
- Disease: RP, Gene: USH2A
- USH2A is too large for single AAV (15.6 kb cDNA vs 4.7 kb AAV capacity)
- Search treatment registry for USH2A-specific therapies
- RAG: retrieve recent publications on USH2A therapeutic approaches

Output: "For USH2A-related RP, the therapeutic landscape is challenging 
because USH2A is too large for conventional single-AAV gene replacement. 
Current approaches include:

1. Antisense oligonucleotide (QR-421a/ultevursen): Targets the common 
   exon 13 splice mutation. Phase 2/3 trial (STELLAR) ongoing.
   
2. Dual-vector gene therapy: Split-intein approaches being developed 
   to deliver the full-length usherin protein.
   
3. Gene-agnostic approaches: OCU400 (NR2E3 modifier) is in Phase 1/2 
   for multiple RP genotypes including USH2A.
   
4. Standard of care: Vitamin A palmitate supplementation may slow 
   progression ~20% per year."
```

### 7.3 Structured Report Output

For clinical integration, the agent can generate structured reports:

```julia
struct ClinicalReport
    patient_id::String
    exam_date::Date
    
    # ERG summary
    erg_quality::Symbol
    erg_interpretation::String
    
    # Model fit summary
    fit_quality::Float64  # R² or equivalent
    key_parameter_findings::Vector{String}
    
    # Diagnostic assessment
    primary_diagnosis::DiagnosisHypothesis
    differential::Vector{DiagnosisHypothesis}
    
    # Staging
    disease_stage::String
    quantitative_severity::Dict{String, Float64}
    
    # Recommendations
    treatments::Vector{TreatmentRecommendation}
    additional_tests::Vector{String}
    follow_up_interval::String
    
    # Disclaimer
    disclaimer::String  # "This report is generated by an AI system..."
end
```

---

## 8. Implementation Roadmap

### Phase 1: Lexicon Foundation (Weeks 1–2)

- Finalize JSON schema and populate all disease entries (current doc covers 13 diseases; expand to ~25)
- Add normative/healthy parameter baselines from literature
- Build disease fingerprint matching algorithm
- Validate fingerprints against published ERG case series

### Phase 2: Parameter Interpreter (Weeks 3–4)

- Implement parameter comparison to healthy baselines
- Build cell-type implication logic
- Implement disease fingerprint cosine similarity matching
- Test with synthetic ERG data (generate ERGs from known disease parameter sets)

### Phase 3: RAG Literature Index (Weeks 5–6)

- Index PubMed corpus (ERG + retinal disease; ~5,000 key papers)
- Build retrieval pipeline with PubMedBERT embeddings
- Implement hybrid search (vector + keyword metadata filters)
- Build ClinicalTrials.gov integration for treatment registry

### Phase 4: Agent Integration (Weeks 7–8)

- Implement DiagnosticAgent struct and query routing
- Build prompt library for each query type
- Implement evidence fusion reasoning chain
- Build structured report generator

### Phase 5: Validation & Iteration (Weeks 9–12)

- Test against published case series with known diagnoses
- Evaluate diagnostic accuracy on held-out cases
- Clinician review of agent reasoning quality
- Iterate on lexicon entries based on edge cases

### Dependencies

| Component | Technology | Notes |
|-----------|------------|-------|
| LLM backbone | Claude API (Anthropic) or local Llama | Claude preferred for reasoning quality |
| Vector DB | FAISS or Qdrant | For RAG literature index |
| Embeddings | PubMedBERT or BioLinkBERT | Domain-specific medical embeddings |
| Structured DB | SQLite or PostgreSQL | For lexicon + treatment registry |
| Frontend | React or Genie.jl | Web-based query interface |
| Model interface | Julia (RetinalTwin package) | Existing digital twin codebase |

---

## Appendix A: Lexicon Expansion Targets

Diseases to add in subsequent iterations:

| Priority | Disease | Category |
|----------|---------|----------|
| High | Cone-Rod Dystrophy | Inherited |
| High | Choroideremia | Inherited |
| High | Wet AMD | Age-related |
| High | Usher Syndrome (Types 1/2/3) | Inherited + syndromic |
| Medium | KCNV2 Retinopathy | Inherited |
| Medium | Bradyopsia | Inherited |
| Medium | Hydroxychloroquine Toxicity | Toxic |
| Medium | Vigabatrin Toxicity | Toxic |
| Medium | Branch Retinal Vein Occlusion | Vascular |
| Medium | Sickle Cell Retinopathy | Vascular |
| Low | Cancer-Associated Retinopathy | Autoimmune |
| Low | Melanoma-Associated Retinopathy | Autoimmune |
| Low | Birdshot Chorioretinopathy | Inflammatory |
| Low | AZOOR | Inflammatory |

## Appendix B: Model Extensions Needed

Several diseases reveal gaps in the current 193-ODE model:

| Gap | Required For | Extension |
|-----|-------------|-----------|
| Cone subtypes (S vs. L/M) | ESC syndrome, achromatopsia | Split cone compartment into S-cone and L/M-cone |
| Cl⁻ conductance in RPE | Best disease | Add bestrophin-1 Cl⁻ channel to RPE model |
| RGC spiking layer | PhNR, glaucoma, DR | Add simple integrate-and-fire RGC |
| Vascular/ischemia module | DR, RVO, ROP | Oxygen supply model modulating cell viability |
| Visual cycle kinetics | Stargardt, RPE65-LCA | Explicit retinoid cycling between OS and RPE |

These extensions should be prioritized based on clinical utility. The RGC layer and visual cycle are highest impact.
