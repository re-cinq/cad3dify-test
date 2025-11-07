# TIF-to-STEP AI Pipeline: Project Plan & Feasibility Assessment

**Repository**: cad3dify-test
**Target**: On-premise deployment with Tesla GPU
**Model**: Qwen3-VL-8B (local inference)
**Approach**: Zero-shot prompting with Human-in-the-Loop (HITL)
**Date**: 2025-11-06

---

## Executive Summary

### Assessment Result: ✅ **HIGHLY SUITABLE BASE**

The `cad3dify` repository provides an **excellent foundation** for the TIF-to-STEP AI pipeline project. It already implements 80% of the required infrastructure:

- ✅ Vision-Language Model integration (LangChain-based)
- ✅ STEP file generation using CadQuery
- ✅ Iterative refinement with visual feedback
- ✅ Streamlit UI and CLI interfaces
- ✅ Proven image-to-3D CAD workflow

### Key Adaptations Required:

1. **Add TIF file support** (trivial - 1 hour)
2. **Integrate Qwen3-VL for on-prem inference** (2-3 days)
3. **Build structured JSON extraction layer** (1 week)
4. **Create HITL validation interface** (1-2 weeks)
5. **Update pipeline for JSON-first workflow** (3-5 days)

### Estimated Timeline:

- **Phase 1 (Feasibility Study)**: 4-6 weeks
- **Phase 2 (Alpha with HITL)**: 3-4 weeks
- **Total to MVP**: 7-10 weeks

---

## 1. Current Repository Analysis

### 1.1 What the Repository Already Has

#### Core Architecture
```
Input Image (2D CAD)
    ↓
VLM Code Generator (GPT-5/Claude/Gemini/Llama)
    ↓
CadQuery Python Code
    ↓
AI-Powered Python Execution + Debugging
    ↓
STEP File
    ↓
Render to PNG
    ↓
Iterative Refinement (compare rendered vs original)
    ↓
Final STEP File
```

#### Key Components

| Component | File | Status | Reusability |
|-----------|------|--------|-------------|
| **Image handling** | `image.py` | ✅ Complete | 95% - just add TIF |
| **VLM integration** | `chat_models.py` | ✅ Multi-provider | 70% - needs local model support |
| **Code generation** | `v1/cad_code_generator.py` | ✅ Proven | 80% - adapt for JSON input |
| **Code execution** | `agents.py` | ✅ With debugging | 100% - perfect as-is |
| **STEP generation** | Uses CadQuery | ✅ Production-ready | 100% - no changes needed |
| **Refinement loop** | `v1/cad_code_refiner.py` | ✅ Visual comparison | 100% - keep as-is |
| **Rendering** | `render.py` | ✅ STEP → PNG | 100% - keep as-is |
| **CLI interface** | `scripts/cli.py` | ✅ Working | 90% - minor tweaks |
| **Web UI** | `scripts/app.py` | ✅ Streamlit | 60% - needs HITL features |

#### Technology Stack (Already In Place)

- **Python 3.10+**
- **LangChain 0.3+**: AI orchestration framework
- **CadQuery 2.4**: Parametric CAD modeling
- **Streamlit 1.37**: Web interface
- **PIL/OpenCV**: Image processing
- **Poetry**: Dependency management

### 1.2 What's Missing for Your Project

| Requirement | Current State | Gap | Effort |
|-------------|--------------|-----|---------|
| **TIF file support** | ❌ Only JPG/PNG/GIF | Need to add TIF/TIFF | 🟢 1 hour |
| **On-prem model** | ❌ Cloud APIs only | Need vLLM/HF integration | 🟡 2-3 days |
| **Qwen3-VL integration** | ❌ Not supported | Add model configuration | 🟡 2-3 days |
| **JSON extraction** | ❌ Direct image→code | Need structured parser | 🔴 1 week |
| **HITL interface** | ❌ Fully automated | Need validation UI | 🔴 1-2 weeks |
| **JSON→Code pipeline** | ❌ Not implemented | Enhance code generator | 🟡 3-5 days |

---

## 2. Proposed Architecture

### 2.1 Modified Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    TIF-to-STEP AI Pipeline                  │
└─────────────────────────────────────────────────────────────┘

Step 1: Technical Drawing Input
┌──────────────┐
│ TIF Drawing  │ (2D technical drawing with dimensions)
└──────┬───────┘
       │
       v
Step 2: VLM Structured Extraction (NEW)
┌─────────────────────────────────────┐
│ TechnicalDrawingParserChain         │
│ - Uses Qwen3-VL-8B (on Tesla GPU)  │
│ - Extracts: dimensions, features,   │
│   tolerances, views, title block    │
│ - Output: Structured JSON           │
└──────┬──────────────────────────────┘
       │
       v
       JSON Data
┌──────────────────────────────────────────────┐
│ {                                            │
│   "title_block": {...},                     │
│   "views": [{dimensions: [...]}],           │
│   "features": [{type: "hole", ...}],        │
│   "tolerances": {...}                       │
│ }                                            │
└──────┬───────────────────────────────────────┘
       │
       v
Step 3: Human-in-the-Loop Validation (NEW)
┌────────────────────────────────────────┐
│ HITL Validation Interface              │
│ - Display: Original TIF | JSON editor  │
│ - Human reviews extracted data         │
│ - Corrects errors (dimensions, units)  │
│ - Adds missing features                │
│ - Clicks "Approve"                     │
└──────┬─────────────────────────────────┘
       │
       v
       Validated JSON
┌──────────────────────────────────────┐
│ Human-corrected structured data      │
└──────┬───────────────────────────────┘
       │
       v
Step 4: Code Generation (ENHANCED)
┌──────────────────────────────────────┐
│ CadCodeGeneratorChain (Modified)     │
│ - Input: TIF image + validated JSON  │
│ - Generates: CadQuery Python code    │
│ - Uses JSON as ground truth          │
└──────┬───────────────────────────────┘
       │
       v
       Python Code
┌──────────────────────────────────────┐
│ import cadquery as cq                │
│ result = cq.Workplane("XY")          │
│   .box(50, 30, 20)  # from JSON      │
│   .faces(">Z").hole(10)              │
│ cq.exporters.export(result, ...)    │
└──────┬───────────────────────────────┘
       │
       v
Step 5: Execution + Debugging (EXISTING - Keep as-is)
┌──────────────────────────────────────┐
│ execute_python_code (AI Agent)       │
│ - Runs code in safe environment      │
│ - Auto-fixes errors (up to 8 tries)  │
│ - Generates STEP file                │
└──────┬───────────────────────────────┘
       │
       v
       STEP File (initial)
┌──────────────────────────────────────┐
│ output.step                          │
└──────┬───────────────────────────────┘
       │
       v
Step 6: Visual Refinement Loop (EXISTING - Keep as-is)
┌──────────────────────────────────────┐
│ For i in range(3):                   │
│   1. Render STEP → PNG               │
│   2. Compare: original TIF vs render │
│   3. CadCodeRefinerChain             │
│   4. Generate refined code           │
│   5. Execute → new STEP              │
└──────┬───────────────────────────────┘
       │
       v
       Final STEP File
┌──────────────────────────────────────┐
│ output.step (refined)                │
│ Ready for CAD software               │
└──────────────────────────────────────┘
```

### 2.2 New Components to Build

#### Component 1: `TechnicalDrawingParserChain`
**Location**: `cad3dify/v1/technical_drawing_parser.py`

```python
class TechnicalDrawingParserChain(SequentialChain):
    """
    Extracts structured JSON from technical drawings.

    Prompt Strategy:
    - Few-shot examples (2-3 annotated drawings)
    - Step-by-step reasoning instructions
    - JSON schema enforcement
    - Anti-hallucination prompts

    Output JSON Schema:
    {
      "title_block": {
        "part_name": str,
        "drawing_number": str,
        "scale": str,
        "material": str,
        "units": "mm" | "inches"
      },
      "views": [
        {
          "view_type": "front" | "top" | "side" | "isometric",
          "dimensions": [
            {
              "type": "linear" | "diameter" | "radius" | "angle",
              "value": float,
              "unit": str,
              "label": str,
              "tolerance": str,
              "location": str
            }
          ]
        }
      ],
      "features": [
        {
          "type": "hole" | "slot" | "pocket" | "chamfer" | "fillet",
          "dimensions": {...},
          "position": str,
          "quantity": int
        }
      ],
      "tolerances": {...},
      "notes": [str]
    }
    """
```

**Prompt Engineering Strategy**:
1. **System prompt**: "You are an expert mechanical engineer analyzing technical drawings"
2. **Few-shot examples**: Include 2-3 perfect TIF→JSON examples
3. **Chain-of-thought**: "First identify views, then extract dimensions, then features..."
4. **JSON validation**: Enforce strict schema with Pydantic
5. **Hallucination prevention**: "If you cannot read a dimension clearly, omit it"

#### Component 2: HITL Validation Interface
**Location**: `scripts/app_hitl.py`

```
┌─────────────────────────────────────────────────────────┐
│           TIF-to-STEP Pipeline (HITL Mode)              │
├─────────────────────────────────────────────────────────┤
│  Sidebar                                                │
│  ┌─────────────┐                                       │
│  │ 1. Upload   │                                       │
│  │ 2. Extract  │ ← Current Step                       │
│  │ 3. Validate │                                       │
│  │ 4. Generate │                                       │
│  └─────────────┘                                       │
├──────────────────────────┬──────────────────────────────┤
│ Original Drawing         │ Extracted JSON               │
│ ┌──────────────────────┐ │ ┌──────────────────────────┐ │
│ │                      │ │ │ {                        │ │
│ │   [TIF IMAGE]        │ │ │   "title_block": {       │ │
│ │                      │ │ │     "part_name": "..."   │ │
│ │   Front View         │ │ │   },                     │ │
│ │                      │ │ │   "views": [             │ │
│ │   50mm ←→            │ │ │     {                    │ │
│ │                      │ │ │       "dimensions": [    │ │
│ └──────────────────────┘ │ │         {"value": 50}    │ │
│                          │ │       ]                  │ │
│                          │ │     }                    │ │
│                          │ │   ]                      │ │
│                          │ │ }                        │ │
│                          │ │                          │ │
│                          │ │ [Edit above if needed]   │ │
│                          │ └──────────────────────────┘ │
│                          │                              │
│                          │ [✓ Approve] [✗ Reject]      │
└──────────────────────────┴──────────────────────────────┘
```

**Features**:
- Side-by-side comparison (TIF vs JSON)
- Inline JSON editor with syntax highlighting
- JSON validation before approval
- Save corrections for future reference
- Workflow state management

#### Component 3: Enhanced Pipeline Function
**Location**: `cad3dify/pipeline.py`

```python
def generate_step_from_technical_drawing(
    image_filepath: str,
    output_filepath: str,
    model_type: str = "qwen",
    enable_json_extraction: bool = True,
    json_data: Optional[dict] = None,
    num_refinements: int = 3
) -> dict:
    """
    New pipeline function supporting JSON extraction and HITL.

    Args:
        image_filepath: Path to TIF drawing
        output_filepath: Path for output STEP file
        model_type: "qwen" for local inference
        enable_json_extraction: If True, extract JSON from image
        json_data: Pre-validated JSON (from HITL workflow)
        num_refinements: Number of refinement iterations

    Returns:
        {
            "extracted_json": {...},
            "step_path": "output.step",
            "refinement_count": 3
        }

    Workflow:
        1. If json_data provided → skip extraction (HITL mode)
        2. Else if enable_json_extraction → run parser
        3. Generate code (with JSON context)
        4. Execute code → STEP file
        5. Refine (visual comparison loop)
    """
```

---

## 3. Implementation Plan

### Phase 1: Foundation (Week 1-2)

#### Week 1: Local Model Setup

**Tasks**:
1. Add TIF support to `image.py`
   - Modify `ImageTypes` to include "tif", "tiff"
   - Test with sample TIF files

2. Integrate vLLM for Qwen3-VL
   - Add `vllm` to dependencies
   - Create new provider in `chat_models.py`
   - Add Qwen3-VL-8B configuration

3. Setup on-prem infrastructure
   - Install CUDA drivers on Tesla GPU server
   - Download Qwen3-VL-8B model (~15GB)
   - Test basic inference

**Deliverables**:
- ✅ TIF files load successfully
- ✅ Qwen3-VL runs on Tesla GPU
- ✅ Baseline test: Can Qwen describe a simple technical drawing?

#### Week 2: JSON Extraction Chain

**Tasks**:
1. Create `TechnicalDrawingParserChain`
   - Design JSON schema (Pydantic models)
   - Write extraction prompt (with few-shot examples)
   - Implement JSON parsing and validation

2. Create test dataset
   - Select 5 simple technical drawings
   - Manually create "ground truth" JSON for each
   - Test extraction accuracy

3. Prompt engineering iteration
   - Test on 5 drawings
   - Measure accuracy (dimension extraction)
   - Refine prompt based on errors

**Deliverables**:
- ✅ Working `TechnicalDrawingParserChain`
- ✅ 5 test drawings with ground truth JSON
- ✅ Baseline metrics: F1-score, hallucination rate

**Success Criteria**:
- F1-score > 70% on dimension extraction
- Hallucination rate < 15%
- Valid JSON 100% of the time

### Phase 2: HITL Interface (Week 3-4)

#### Week 3: Basic HITL UI

**Tasks**:
1. Create `app_hitl.py` Streamlit interface
   - File uploader (TIF/PNG/JPG)
   - Two-column layout (image | JSON)
   - JSON text editor with validation
   - Approval buttons

2. Implement workflow state machine
   - Step 1: Upload
   - Step 2: Extract
   - Step 3: Validate
   - Step 4: Generate

3. Connect to pipeline
   - Call `TechnicalDrawingParserChain`
   - Pass validated JSON to code generator
   - Display results

**Deliverables**:
- ✅ Working HITL Streamlit app
- ✅ User can upload, review, correct, approve

#### Week 4: Pipeline Integration

**Tasks**:
1. Create `generate_step_from_technical_drawing()` function
   - Support both auto and HITL modes
   - Handle JSON input
   - Fallback to direct image→code if extraction fails

2. Enhance `CadCodeGeneratorChain` (optional)
   - Add JSON context to prompt
   - Use JSON dimensions as constraints

3. End-to-end testing
   - Test complete flow: TIF → JSON → HITL → STEP
   - Test on 10 different drawings
   - Measure time per drawing

**Deliverables**:
- ✅ Full pipeline working end-to-end
- ✅ 10 test cases documented
- ✅ Performance metrics captured

**Success Criteria**:
- 100% of drawings produce valid STEP files
- HITL validation takes < 2 minutes per drawing
- Generated STEP files match dimensions (within tolerance)

### Phase 3: Feasibility Gate (Week 5-6)

#### Week 5: Validation & Metrics

**Tasks**:
1. Build evaluation framework
   - Dimension accuracy: Compare JSON dimensions to ground truth
   - STEP validation: Measure dimensions in CAD software
   - Hallucination detection: Count invented features

2. Test on validation set
   - Run on 10-15 diverse drawings
   - Measure KPIs (see below)
   - Document failure modes

3. Error analysis
   - Categorize errors (dimension misread, feature missed, etc.)
   - Identify prompt improvements
   - Document limitations

**KPIs to Measure**:
| KPI | Target (Go/No-Go) | How to Measure |
|-----|-------------------|----------------|
| Dimension F1-score | > 85% | Compare extracted JSON to ground truth |
| Hallucination rate | < 10% | Count dimensions/features not in drawing |
| STEP accuracy | > 90% | Measure dimensions in CAD software |
| HITL correction time | < 3 min/drawing | Time human validation |
| End-to-end time | < 5 min/drawing | Total pipeline time |

#### Week 6: Decision Gate

**Tasks**:
1. Compile feasibility report
   - Summary of results
   - KPI scorecard
   - Risk assessment
   - Recommendation (GO / PIVOT / STOP)

2. Stakeholder presentation
   - Demo the HITL workflow
   - Show sample STEP files
   - Present metrics

3. Document decision rationale

**Possible Outcomes**:

**✅ GO** (KPIs met):
- Proceed to Phase 4 (Beta with engineers)
- Plan MLOps infrastructure (future)

**⚠️ PIVOT** (Partial success):
- Keep HITL, skip automation
- Focus on "AI-assisted" workflow
- Human creates JSON, AI generates code

**❌ STOP** (KPIs missed):
- Technology not mature enough
- Reassess in 6-12 months

---

## 4. Technical Specifications

### 4.1 On-Premise Infrastructure

#### Hardware Requirements
```
Server Specification:
- GPU: NVIDIA Tesla V100 (16GB VRAM) or better
- CPU: 16 cores minimum
- RAM: 64 GB minimum
- Storage: 500 GB SSD (for model cache + data)
- Network: 10 Gbps (for model downloads)
```

#### Software Stack
```
OS: Ubuntu 22.04 LTS
Python: 3.10 or 3.11
CUDA: 12.1+
Driver: 525.xx or later

Key Libraries:
- vllm: 0.6.0+ (high-performance inference)
- transformers: 4.40.0+
- torch: 2.2.0+ with CUDA
- langchain: 0.3.18+
- cadquery: 2.4.0
- streamlit: 1.37+
```

#### Model Deployment Options

**Option A: vLLM Server (Recommended)**
```bash
# Start vLLM server
vllm serve Qwen/Qwen2-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9

# Benefits:
# - Highest throughput (batching, continuous batching)
# - Lowest latency (<100ms per request)
# - Easy to scale (multiple GPUs)
```

**Option B: Direct HuggingFace**
```python
# Load model in-process
model = AutoModel.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    device_map="auto"  # Auto GPU
)

# Benefits:
# - Simpler setup (no server)
# - Good for single-user
```

### 4.2 Prompt Engineering Strategy

#### JSON Extraction Prompt Structure

```
SYSTEM PROMPT:
You are an expert mechanical engineer specializing in reading technical
drawings. Your task is to extract structured information from engineering
drawings with extreme precision.

INSTRUCTIONS:
1. Carefully examine ALL views (front, top, side, isometric)
2. Read EVERY dimension callout with its value and unit
3. Identify geometric features (holes, slots, pockets, etc.)
4. Extract tolerances and specifications
5. Output ONLY valid JSON in the specified schema

CRITICAL RULES:
- Be precise: Extract exact numbers, never estimate
- Include units: Always specify mm, inches, degrees
- Don't hallucinate: If unclear, omit rather than guess
- Be thorough: Extract ALL dimensions and features

FEW-SHOT EXAMPLES:
[Example 1: Simple block with hole]
Input: [TIF image]
Output: {"title_block": {...}, "views": [...], "features": [...]}

[Example 2: Complex bracket]
Input: [TIF image]
Output: {...}

NOW ANALYZE THIS DRAWING:
[User's TIF image]

OUTPUT (JSON only, no other text):
```

#### Anti-Hallucination Techniques
1. **Explicit uncertainty handling**: "If you cannot read a dimension, omit it"
2. **Schema validation**: Pydantic models enforce structure
3. **Confidence scoring**: (Future) Ask model to rate confidence per dimension
4. **HITL as safety net**: Human catches all errors

### 4.3 HITL Interface Specifications

#### UI/UX Requirements
```
Layout: Two-column (image | JSON editor)
Technology: Streamlit (already in use)
Responsiveness: Desktop-optimized (1920x1080+)

Features:
- Image viewer: Pan, zoom, rotate TIF
- JSON editor: Syntax highlighting, collapsible sections
- Validation: Real-time JSON syntax check
- Diff view: Highlight AI changes vs manual corrections
- Keyboard shortcuts: Ctrl+Enter to approve

Workflow:
1. Upload TIF
2. AI extracts (loading spinner)
3. Display side-by-side
4. Human edits JSON
5. Click "Approve" → generate STEP
6. Download STEP file
```

#### State Management
```python
# Session state variables
st.session_state = {
    "step": "upload",  # upload | extract | validate | generate
    "image_path": None,
    "extracted_json": None,
    "validated_json": None,
    "step_path": None
}
```

### 4.4 JSON Schema (Pydantic Models)

```python
from pydantic import BaseModel, Field
from typing import Literal, List, Optional

class TitleBlock(BaseModel):
    part_name: str
    drawing_number: Optional[str] = None
    scale: Optional[str] = None
    material: Optional[str] = None
    units: Literal["mm", "inches"]

class Dimension(BaseModel):
    type: Literal["linear", "diameter", "radius", "angle"]
    value: float
    unit: str
    label: Optional[str] = None
    tolerance: Optional[str] = None
    location: str  # Description of where this dimension is

class View(BaseModel):
    view_type: Literal["front", "top", "side", "isometric", "section"]
    dimensions: List[Dimension] = []

class Feature(BaseModel):
    type: Literal["hole", "slot", "pocket", "chamfer", "fillet",
                  "counterbore", "countersink", "thread"]
    dimensions: dict
    position: str
    quantity: int = 1
    notes: Optional[str] = None

class Tolerances(BaseModel):
    general_linear: Optional[str] = None
    general_angular: Optional[str] = None
    surface_finish: Optional[str] = None
    notes: Optional[str] = None

class TechnicalDrawingData(BaseModel):
    title_block: TitleBlock
    views: List[View]
    features: List[Feature] = []
    tolerances: Tolerances
    notes: List[str] = []
```

---

## 5. Risk Assessment

### 5.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Qwen3-VL cannot read dimensions accurately** | Medium | High | Use HITL as fallback; test early in Week 1 |
| **TIF files have poor quality** | Low | Medium | Preprocessing: deskew, enhance contrast |
| **GPU runs out of memory** | Low | Medium | Reduce batch size; use 7B instead of 8B model |
| **STEP generation fails** | Low | High | Existing pipeline handles this well |
| **JSON extraction is too slow** | Medium | Low | Optimize prompts; use vLLM for speed |
| **Complex drawings exceed context window** | Medium | Medium | Split into regions; process views separately |

### 5.2 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Prompt engineering takes longer than expected** | High | Medium | Budget 2 weeks for iteration; start early |
| **HITL workflow is too slow** | Medium | Medium | Optimize UI; train users; consider batch mode |
| **KPIs not met at gate** | Medium | High | PIVOT to HITL-first (still valuable) |
| **User adoption is low** | Low | High | Involve engineers early; show value quickly |

### 5.3 Dependency Risks

| Dependency | Risk | Mitigation |
|------------|------|------------|
| **Qwen3-VL model availability** | Model deprecated/removed | Cache locally; test alternatives (LLaVA, CogVLM) |
| **vLLM compatibility** | Breaking changes | Pin versions; test before upgrade |
| **CadQuery updates** | API changes | Pin to 2.4.0; test before upgrade |
| **LangChain updates** | Breaking changes | Use ^0.3.18; monitor changelogs |

---

## 6. Success Metrics

### 6.1 Phase 1 KPIs (Feasibility Gate)

| Metric | Measurement | Target | Stretch Goal |
|--------|-------------|--------|--------------|
| **Dimension Extraction F1** | Compare to ground truth | > 85% | > 90% |
| **Hallucination Rate** | Count invented dimensions | < 10% | < 5% |
| **JSON Validity** | Parse rate | 100% | 100% |
| **STEP Accuracy** | Measure in CAD | > 90% | > 95% |
| **Processing Time** | End-to-end per drawing | < 5 min | < 3 min |
| **HITL Correction Time** | Human validation | < 3 min | < 2 min |

### 6.2 User Satisfaction (Post-HITL)

- **Ease of Use**: Can engineer use without training? (Target: Yes)
- **Trust**: Do engineers trust the output? (Target: >80% say yes)
- **Value**: Does it save time vs manual CAD? (Target: >50% time savings)

---

## 7. Cost Estimate

### 7.1 Development Time

| Phase | Duration | Engineer-Weeks |
|-------|----------|----------------|
| Phase 1: Foundation | 2 weeks | 2 |
| Phase 2: HITL Interface | 2 weeks | 2 |
| Phase 3: Validation & Gate | 2 weeks | 2 |
| **Total to Feasibility Gate** | **6 weeks** | **6** |
| Phase 4 (if GO): Beta deployment | 4 weeks | 4 |
| **Total to Beta** | **10 weeks** | **10** |

### 7.2 Infrastructure Costs (On-Prem)

| Item | One-Time | Annual |
|------|----------|--------|
| **Server** (if new) | $10,000-$15,000 | - |
| **GPU** (Tesla V100) | Included in server | - |
| **Storage** | $1,000 | - |
| **Power & Cooling** | - | $2,000 |
| **Maintenance** | - | $1,000 |
| **Total** | **$11,000-$16,000** | **$3,000** |

**Note**: If Tesla GPU already available, one-time cost is near-zero.

### 7.3 Software Costs

| Item | Cost |
|------|------|
| **Open-source stack** (Python, vLLM, Qwen, CadQuery) | $0 |
| **Cloud API fallback** (optional, for comparison) | ~$50/month |
| **CAD software licenses** (for validation) | Existing |
| **Total** | **$0-$50/month** |

---

## 8. Go/No-Go Decision Criteria

### Week 6 Decision Gate

#### ✅ **GO** Criteria (Proceed to Beta)
- ✅ Dimension F1-score > 85%
- ✅ Hallucination rate < 10%
- ✅ STEP files import in CAD software
- ✅ Engineers find HITL workflow acceptable (<3 min)
- ✅ At least 8/10 test drawings produce correct STEP

**Action**: Proceed to Phase 4 (Beta deployment with 3-5 engineers)

#### ⚠️ **PIVOT** Criteria (Change Approach)
- ⚠️ F1-score 70-85% (acceptable but needs improvement)
- ⚠️ Hallucination rate 10-15% (manageable with HITL)
- ⚠️ Code generation works but JSON extraction weak

**Action**:
1. Make HITL primary workflow (human creates JSON manually)
2. Use AI only for code generation
3. Still valuable: JSON template → Code → STEP

#### ❌ **STOP** Criteria (Technology Not Ready)
- ❌ F1-score < 70%
- ❌ Hallucination rate > 15%
- ❌ STEP files consistently wrong
- ❌ Processing time > 10 min per drawing
- ❌ Engineers reject the workflow

**Action**: Pause project; reassess in 6-12 months when models improve

---

## 9. Next Steps (Immediate Actions)

### If Approved to Proceed:

1. **This Week**:
   - [ ] Confirm Tesla GPU server access
   - [ ] Install CUDA drivers and test GPU
   - [ ] Download Qwen3-VL-8B model
   - [ ] Clone cad3dify repository

2. **Week 1**:
   - [ ] Add TIF support (1 hour)
   - [ ] Integrate vLLM provider (2 days)
   - [ ] Test basic inference (1 day)
   - [ ] Collect 5 sample technical drawings

3. **Week 2**:
   - [ ] Build `TechnicalDrawingParserChain`
   - [ ] Create ground truth JSON for 5 drawings
   - [ ] Measure baseline metrics
   - [ ] Review & decide whether to continue

---

## 10. Appendices

### Appendix A: File Structure After Implementation

```
cad3dify-test/
├── cad3dify/
│   ├── __init__.py
│   ├── image.py                    # MODIFIED: Add TIF support
│   ├── chat_models.py              # MODIFIED: Add vLLM provider
│   ├── pipeline.py                 # MODIFIED: Add new pipeline function
│   ├── agents.py                   # KEEP AS-IS
│   ├── render.py                   # KEEP AS-IS
│   └── v1/
│       ├── cad_code_generator.py   # KEEP AS-IS (or minor enhancement)
│       ├── cad_code_refiner.py     # KEEP AS-IS
│       └── technical_drawing_parser.py  # NEW
├── scripts/
│   ├── cli.py                      # MINOR UPDATE: Add --enable-hitl flag
│   ├── app.py                      # MINOR UPDATE: Add TIF to uploader
│   └── app_hitl.py                 # NEW: HITL interface
├── tests/                          # NEW: Test suite
│   ├── test_parser.py
│   ├── test_pipeline.py
│   └── fixtures/
│       └── sample_drawings/
├── docs/
│   ├── DEPLOYMENT_ONPREM.md        # NEW: Deployment guide
│   └── QUICKSTART.md               # NEW: Quick start
├── pyproject.toml                  # MODIFIED: Add vllm, langchain-community
├── README.md                       # UPDATE: Add TIF-to-STEP section
└── .env.sample                     # NEW: Environment variables template
```

### Appendix B: Sample Test Drawings Needed

Collect 5-15 drawings covering:
1. **Simple**: Rectangular block with 1-2 holes
2. **Medium**: Bracket with slots and chamfers
3. **Medium**: Cylindrical shaft with keyway
4. **Complex**: Multi-view part with threads
5. **Complex**: Assembly drawing (stretch goal)

Each should have:
- Clear dimension callouts
- Multiple views (front, top, side)
- Standard tolerances
- Title block

### Appendix C: Alternative Models (Backup Options)

If Qwen3-VL doesn't work:
1. **LLaVA-Next** (13B): Open-source, good at technical images
2. **CogVLM** (17B): Strong vision capabilities
3. **Phi-3-Vision** (4B): Smaller, faster, lower accuracy
4. **Gemini Pro Vision** (Cloud): Fallback if on-prem fails

### Appendix D: References

- **CadQuery Docs**: https://cadquery.readthedocs.io/
- **vLLM Docs**: https://docs.vllm.ai/
- **Qwen Model Card**: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
- **LangChain Docs**: https://python.langchain.com/
- **STEP File Format**: ISO 10303-21

---

## Conclusion

The `cad3dify` repository is an **excellent base** for your TIF-to-STEP AI pipeline. It provides:
- ✅ 80% of required infrastructure
- ✅ Proven STEP generation capabilities
- ✅ Extensible architecture
- ✅ Active codebase

**Key adaptations** are straightforward and low-risk:
- TIF support: 1 hour
- Qwen3-VL integration: 2-3 days
- JSON extraction: 1 week
- HITL interface: 1-2 weeks

**Estimated timeline to feasibility gate**: 6 weeks
**Estimated timeline to working beta**: 10 weeks

**Recommendation**: **Proceed with Phase 1** (2-week sprint to test viability)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-06
**Author**: Claude (AI Assistant)
**Status**: READY FOR REVIEW
