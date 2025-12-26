# components/orchestration.py

"""
Component 5: Agent Orchestration System
Technology: LangGraph

Enhanced Orchestration with:
- Per-agent error isolation (agent failures don't affect others)
- Parallel agent execution
- ScoreCard + charts_data generation
- Optional design comparison node
- Comprehensive FullReport building
"""

from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime
import json
import traceback
import uuid

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback - LangGraph not available
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    START = None
    END = None

from components.models import (
    DesignInput, AgentResult, ScoreCard, FullReport,
    ComparisonResult, VisualFeedback, RAGCitation
)
from components.rag_system import retrieve_relevant_patterns
from components.agents_real import (
    analyze_ux_design,
    analyze_market_positioning,
    analyze_conversion_optimization,
    analyze_brand_consistency_check,
    analyze_visual_design
)
from components.agents import compare_designs


# ============================================================================
# Enhanced State Definition
# ============================================================================

class AnalysisState(TypedDict):
    """
    Enhanced analysis state schema with new models
    """
    # Inputs
    design_inputs: List[DesignInput]
    enabled_agents: List[str]
    analysis_mode: str  # "single" or "compare"
    comparison_goal: Optional[str]  # "overall", "conversion", "brand", "ux"
    platform: str
    creative_type: str
    api_key: Optional[str]  # BYOK: User-provided API key

    # Processing
    preprocessed_images: Dict[str, Any]
    rag_patterns: List[Dict[str, Any]]
    rag_citations: List[RAGCitation]

    # Results
    agent_results: List[AgentResult]
    comparison_result: Optional[ComparisonResult]
    scores: Optional[ScoreCard]
    charts_data: Dict[str, Any]
    visual_feedback: Optional[VisualFeedback]

    # Metadata
    full_report: Optional[FullReport]
    errors: List[str]
    execution_times: Dict[str, float]

    # Legacy compatibility
    image_base64: str
    image_embedding: list
    image_metadata: dict
    model_used: str
    top_k: int

    # Legacy outputs (for compatibility)
    visual_analysis: dict
    ux_analysis: dict
    market_analysis: dict
    conversion_analysis: dict
    brand_analysis: dict

    # Final output
    final_report: dict
    error: str

    # Progress tracking
    current_step: int
    total_steps: int
    step_message: str


# ============================================================================
# Node 1: Preprocess Images
# ============================================================================

def preprocess_images_node(state: AnalysisState) -> AnalysisState:
    """
    Node 1: Load, validate, and preprocess images.

    - Validate image formats
    - Extract metadata (dimensions, color space)
    - Prepare for analysis
    """
    print("\n[Node 1/6] Preprocessing images...")
    start_time = datetime.now()

    try:
        preprocessed = {}

        for design in state.get("design_inputs", []):
            design_id = design.id

            # Validate image exists
            if not design.image_base64:
                raise ValueError(f"Design {design_id}: No image data provided")

            preprocessed[design_id] = {
                "design_id": design_id,
                "filename": design.filename,
                "platform": design.platform,
                "creative_type": design.creative_type,
                "metadata": design.metadata or {},
                "image_base64": design.image_base64,
            }

        state["preprocessed_images"] = preprocessed

    except Exception as e:
        error_msg = f"Image preprocessing failed: {str(e)}"
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_msg)
        print(f"  ✗ {error_msg}")

    elapsed = (datetime.now() - start_time).total_seconds()
    state["execution_times"]["preprocess"] = elapsed
    print(
        f"  ✓ Preprocessed {len(state.get('preprocessed_images', {}))} images in {elapsed:.2f}s")

    return state


# ============================================================================
# Node 2: RAG Retrieval
# ============================================================================

def retrieve_rag_patterns_node(state: AnalysisState) -> AnalysisState:
    """
    Node 2: Retrieve relevant design patterns from RAG.

    - Query RAG system once per request
    - Retrieve top-k design patterns
    - Create citations for usage
    """
    print("\n[Node 2/6] Retrieving RAG patterns...")
    start_time = datetime.now()

    try:
        # Build RAG query from design characteristics
        platform = state.get("platform", "Instagram")
        creative_type = state.get("creative_type", "Marketing Creative")
        query = f"platform: {platform}, type: {creative_type}, design patterns"

        # Retrieve patterns (once per request)
        patterns = retrieve_relevant_patterns(query, top_k=5)

        state["rag_patterns"] = patterns

        # Create citation objects
        citations = []
        for pattern in patterns:
            citation = RAGCitation(
                pattern_id=pattern.get("id", "unknown"),
                pattern_title=pattern.get("title", "Unknown Pattern"),
                category=pattern.get("category", "general"),
                relevance_score=pattern.get("relevance_score", 0.5),
                used_by=[]
            )
            citations.append(citation)

        state["rag_citations"] = citations

    except Exception as e:
        error_msg = f"RAG retrieval failed: {str(e)}"
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_msg)
        print(f"  ✗ {error_msg}")
        state["rag_patterns"] = []
        state["rag_citations"] = []

    elapsed = (datetime.now() - start_time).total_seconds()
    state["execution_times"]["rag"] = elapsed
    print(
        f"  ✓ Retrieved {len(state.get('rag_patterns', []))} patterns in {elapsed:.2f}s")

    return state


# ============================================================================
# Node 3: Run Agents in Parallel (with error isolation)
# ============================================================================

def run_agent_with_error_isolation(agent_name: str, state: AnalysisState) -> AgentResult:
    """
    Run a single agent with error isolation.

    If agent fails, returns AgentResult with errors field populated,
    so other agents can still run.
    
    IMPORTANT: Each agent gets:
    - Image data (image_base64) for vision analysis
    - API key (from BYOK session or environment)
    - Platform and creative type context
    """
    try:
        print(f"    → {agent_name:25s}", end=" ", flush=True)
        start_time = datetime.now()

        platform = state.get("platform", "Instagram")
        creative_type = state.get("creative_type", "Marketing Creative")
        api_key = state.get("api_key")  # BYOK: User-provided API key
        rag_citations = state.get("rag_citations", [])

        # Get first design input to get image base64
        design_inputs = state.get("design_inputs", [])
        if not design_inputs:
            raise ValueError("No design inputs provided")

        # Use first image for single design analysis (per-image context preserved in app.py)
        first_design = design_inputs[0]
        image_base64 = first_design.image_base64

        # Call appropriate agent function with image data and API key
        if agent_name == "ux_critique":
            result = analyze_ux_design(
                image_base64=image_base64,
                platform=platform,
                creative_type=creative_type,
                api_key=api_key,
                rag_citations=rag_citations
            )
        elif agent_name == "market_research":
            result = analyze_market_positioning(
                image_base64=image_base64,
                platform=platform,
                creative_type=creative_type,
                api_key=api_key,
                rag_citations=rag_citations
            )
        elif agent_name == "conversion_cta":
            result = analyze_conversion_optimization(
                image_base64=image_base64,
                platform=platform,
                creative_type=creative_type,
                api_key=api_key,
                rag_citations=rag_citations
            )
        elif agent_name == "brand_consistency":
            result = analyze_brand_consistency_check(
                image_base64=image_base64,
                platform=platform,
                creative_type=creative_type,
                api_key=api_key,
                rag_citations=rag_citations
            )
        elif agent_name == "visual_analysis":
            result = analyze_visual_design(
                image_base64=image_base64,
                platform=platform,
                creative_type=creative_type,
                api_key=api_key,
                rag_citations=rag_citations
            )
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✓ ({elapsed:.2f}s)")
        return result

    except Exception as e:
        # Return AgentResult with error captured - critical for error isolation
        error_msg = f"{agent_name} agent failed"
        error_traceback = traceback.format_exc()
        print(f"✗ ({error_msg})")

        return AgentResult(
            agent_name=agent_name,
            summary="Agent execution failed",
            findings=[],
            score=0.0,
            subscores={},
            errors=[error_msg, error_traceback],
            latency_ms=0.0,
            tokens_used=None
        )


def run_agents_parallel_node(state: AnalysisState) -> AnalysisState:
    """
    Node 3: Run selected agents in parallel with error isolation.

    - Execute each enabled agent
    - Isolate failures (one agent failure doesn't affect others)
    - Capture errors in AgentResult.errors
    - Continue execution even if agents fail
    """
    print("\n[Node 3/6] Running agents in parallel...")
    start_time = datetime.now()

    agent_results = []
    enabled_agents = state.get("enabled_agents", [])

    if enabled_agents:
        print(f"  Enabled agents: {enabled_agents}")

    # Run each agent with error isolation
    for agent_name in enabled_agents:
        result = run_agent_with_error_isolation(agent_name, state)
        agent_results.append(result)

    state["agent_results"] = agent_results

    # Count successes and failures
    successful = len([r for r in agent_results if not r.errors])
    failed = len(agent_results) - successful

    elapsed = (datetime.now() - start_time).total_seconds()
    state["execution_times"]["agents"] = elapsed

    status = f"  ✓ Completed {successful}/{len(agent_results)} agents in {elapsed:.2f}s"
    if failed > 0:
        status += f" ({failed} failed but isolated)"
    print(status)

    return state


# ============================================================================
# Node 4: Aggregate Scores and Prepare Charts
# ============================================================================

def aggregate_scores_node(state: AnalysisState) -> AnalysisState:
    """
    Node 4: Calculate ScoreCard and prepare charts_data.

    - Extract scores from all agent results
    - Calculate weighted overall score
    - Prepare chart data for frontend visualization
    - Skip agents with errors when scoring
    """
    print("\n[Node 4/6] Aggregating scores and preparing charts...")
    start_time = datetime.now()

    try:
        # Map agent names to score fields
        agent_score_map = {
            "visual_analysis": "visual",
            "ux_critique": "ux",
            "market_research": "market",
            "conversion_cta": "conversion",
            "brand_consistency": "brand"
        }

        # Extract scores from agent results
        scores_dict = {
            "visual": 0.0,
            "ux": 0.0,
            "market": 0.0,
            "conversion": 0.0,
            "brand": 0.0
        }

        score_count = 0
        for result in state.get("agent_results", []):
            # Skip agents with errors for scoring
            if not result.errors:
                field = agent_score_map.get(result.agent_name)
                if field:
                    scores_dict[field] = result.score
                    score_count += 1

        # Calculate weighted overall score
        if score_count > 0:
            overall = (
                scores_dict["visual"] * 0.20 +
                scores_dict["ux"] * 0.20 +
                scores_dict["market"] * 0.15 +
                scores_dict["conversion"] * 0.25 +
                scores_dict["brand"] * 0.20
            )
        else:
            overall = 0.0

        # Create ScoreCard
        state["scores"] = ScoreCard(
            overall=overall,
            visual=scores_dict["visual"],
            ux=scores_dict["ux"],
            market=scores_dict["market"],
            conversion=scores_dict["conversion"],
            brand=scores_dict["brand"]
        )

        # Prepare charts data for frontend
        charts_data = {
            "scorecard": {
                "overall": overall,
                "visual": scores_dict["visual"],
                "ux": scores_dict["ux"],
                "market": scores_dict["market"],
                "conversion": scores_dict["conversion"],
                "brand": scores_dict["brand"],
                "timestamp": datetime.utcnow().isoformat()
            },
            "radar_chart": {
                "categories": ["Visual", "UX", "Market", "Conversion", "Brand"],
                "values": [
                    scores_dict["visual"],
                    scores_dict["ux"],
                    scores_dict["market"],
                    scores_dict["conversion"],
                    scores_dict["brand"]
                ]
            },
            "findings_summary": {
                "total_findings": sum(
                    len(r.findings) for r in state.get("agent_results", [])
                    if not r.errors
                ),
                "critical_count": sum(
                    len([f for f in r.findings if f.severity == "critical"])
                    for r in state.get("agent_results", [])
                    if not r.errors
                ),
                "warning_count": sum(
                    len([f for f in r.findings if f.severity == "warning"])
                    for r in state.get("agent_results", [])
                    if not r.errors
                ),
                "info_count": sum(
                    len([f for f in r.findings if f.severity == "info"])
                    for r in state.get("agent_results", [])
                    if not r.errors
                )
            },
            "agent_status": {
                result.agent_name: "success" if not result.errors else "error"
                for result in state.get("agent_results", [])
            }
        }

        state["charts_data"] = charts_data

    except Exception as e:
        error_msg = f"Score aggregation failed: {str(e)}"
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_msg)
        print(f"  ✗ {error_msg}")
        # Create fallback ScoreCard
        state["scores"] = ScoreCard(
            overall=0.0,
            visual=0.0,
            ux=0.0,
            market=0.0,
            conversion=0.0,
            brand=0.0
        )

    elapsed = (datetime.now() - start_time).total_seconds()
    state["execution_times"]["scoring"] = elapsed
    overall = state.get("scores").overall if state.get("scores") else 0.0
    print(
        f"  ✓ Aggregated scores (overall: {overall:.1f}/100) in {elapsed:.2f}s")

    return state


# ============================================================================
# Node 5: Optional Design Comparison
# ============================================================================

def compare_designs_node(state: AnalysisState) -> AnalysisState:
    """
    Node 5: Optional comparison node.

    - Only executed if analysis_mode == "compare"
    - Runs compare_designs agent with collected results
    - Stores comparison_result in state
    """
    print("\n[Node 5/6] Running design comparison (if applicable)...")
    start_time = datetime.now()

    # Only run in compare mode with 2+ designs
    design_inputs = state.get("design_inputs", [])
    if state.get("analysis_mode") != "compare" or len(design_inputs) < 2:
        print("  ⊘ Not in compare mode or insufficient designs")
        state["execution_times"]["compare"] = 0.0
        return state

    try:
        # Organize agent results by design_id
        agent_results_dict = {
            design_inputs[0].id: state.get("agent_results", [])
        }

        # Run comparison
        comparison_result = compare_designs(
            design_inputs=design_inputs,
            agent_results=agent_results_dict,
            comparison_goal=state.get("comparison_goal", "overall")
        )

        state["comparison_result"] = comparison_result

    except Exception as e:
        error_msg = f"Design comparison failed: {str(e)}"
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_msg)
        print(f"  ✗ {error_msg}")
        state["comparison_result"] = None

    elapsed = (datetime.now() - start_time).total_seconds()
    state["execution_times"]["compare"] = elapsed

    if state.get("comparison_result"):
        print(f"  ✓ Comparison completed in {elapsed:.2f}s")

    return state


# ============================================================================
# Node 6: Build Full Report
# ============================================================================

def build_full_report_node(state: AnalysisState) -> AnalysisState:
    """
    Node 6: Assemble complete FullReport.

    - Combine all results into single FullReport object
    - Include scores, agent results, comparison results
    - Add metadata and citations
    """
    print("\n[Node 6/6] Building full report...")
    start_time = datetime.now()

    try:
        # Generate report ID
        report_id = f"report_{uuid.uuid4().hex[:8]}"

        # Create FullReport
        report = FullReport(
            report_id=report_id,
            created_at=datetime.utcnow(),
            app_version="0.2.0",

            # Inputs
            design_inputs=state.get("design_inputs", []),
            analysis_mode=state.get("analysis_mode", "single"),

            # Results
            scores=state.get("scores") or ScoreCard(
                overall=0.0, visual=0.0, ux=0.0,
                market=0.0, conversion=0.0, brand=0.0
            ),
            agent_results=state.get("agent_results", []),
            comparison_result=state.get("comparison_result"),

            # Visual feedback
            visual_feedback=state.get("visual_feedback") or VisualFeedback(),

            # RAG context
            rag_citations=state.get("rag_citations", []),
            rag_top_k=len(state.get("rag_patterns", [])),

            # Metadata
            platform=state.get("platform", "Unknown"),
            creative_type=state.get("creative_type", "Unknown"),
            enabled_agents=state.get("enabled_agents", []),

            # Charts data
            charts_data=state.get("charts_data", {})
        )

        state["full_report"] = report

    except Exception as e:
        error_msg = f"Report building failed: {str(e)}"
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_msg)
        print(f"  ✗ {error_msg}")

    elapsed = (datetime.now() - start_time).total_seconds()
    state["execution_times"]["report"] = elapsed
    print(f"  ✓ Report built in {elapsed:.2f}s")

    return state


# ============================================================================
# Public API
# ============================================================================

def run_analysis_workflow(
    design_inputs: List[DesignInput],
    enabled_agents: List[str],
    analysis_mode: str = "single",
    comparison_goal: Optional[str] = None,
    platform: str = "Instagram",
    creative_type: str = "Marketing Creative",
    api_key: Optional[str] = None
) -> FullReport:
    """
    Run the complete analysis workflow with LangGraph orchestration.

    Args:
        design_inputs: List of design images to analyze
        enabled_agents: List of agent names to run
        analysis_mode: "single" or "compare"
        comparison_goal: "overall", "conversion", "brand", or "ux" (compare mode)
        platform: Target platform
        creative_type: Type of creative
        api_key: Optional API key (BYOK: Bring Your Own Key). If provided, overrides env var

    Returns:
        FullReport with complete analysis results

    Error Handling:
        - Per-agent errors are isolated and captured in AgentResult.errors
        - One agent failure doesn't affect others
        - Workflow continues even if some agents fail
        - Top-level errors captured in state["errors"]
    """
    # Initialize state
    initial_state: AnalysisState = {
        "design_inputs": design_inputs,
        "enabled_agents": enabled_agents,
        "analysis_mode": analysis_mode,
        "comparison_goal": comparison_goal,
        "platform": platform,
        "creative_type": creative_type,
        "api_key": api_key,  # BYOK: Pass user-provided API key

        "preprocessed_images": {},
        "rag_patterns": [],
        "rag_citations": [],

        "agent_results": [],
        "comparison_result": None,
        "scores": None,
        "charts_data": {},
        "visual_feedback": None,

        "full_report": None,
        "errors": [],
        "execution_times": {},

        # Legacy compatibility
        "image_base64": "",
        "image_embedding": [],
        "image_metadata": {},
        "model_used": "gpt-4v",
        "top_k": 5,

        "visual_analysis": {},
        "ux_analysis": {},
        "market_analysis": {},
        "conversion_analysis": {},
        "brand_analysis": {},

        "final_report": {},
        "error": "",

        "current_step": 0,
        "total_steps": 6,
        "step_message": ""
    }

    # Create and run graph
    graph = create_langgraph_workflow()

    print("\n" + "="*70)
    print("DESIGN ANALYSIS WORKFLOW STARTED")
    print("="*70)

    try:
        # Invoke graph
        final_state = graph.invoke(initial_state)
    except Exception as e:
        print(f"\n❌ Workflow execution failed: {str(e)}")
        final_state = initial_state
        final_state["errors"].append(str(e))

    # Print execution summary
    print("\n" + "="*70)
    print("WORKFLOW COMPLETED")
    print("="*70)

    print("\nExecution Times:")
    execution_times = final_state.get("execution_times", {})
    for node, elapsed in execution_times.items():
        print(f"  {node:15s}: {elapsed:7.2f}s")

    total_time = sum(execution_times.values())
    print(f"  {'Total':15s}: {total_time:7.2f}s")

    errors = final_state.get("errors", [])
    if errors:
        print(f"\nTop-Level Errors: {len(errors)}")
        for error in errors:
            print(f"  - {error[:80]}")

    return final_state.get("full_report")


def create_langgraph_workflow():
    """
    Create the LangGraph workflow with 6 nodes.

    Nodes:
    1. preprocess_images: Load and validate images
    2. retrieve_rag_patterns: RAG retrieval (once per request)
    3. run_agents_parallel: Execute selected agents in parallel with error isolation
    4. aggregate_scores: Calculate ScoreCard and prepare charts_data
    5. compare_designs_node: Optional comparison (if compare mode)
    6. build_full_report: Assemble complete FullReport
    """
    graph = StateGraph(AnalysisState)

    # Add nodes
    graph.add_node("preprocess", preprocess_images_node)
    graph.add_node("rag", retrieve_rag_patterns_node)
    graph.add_node("agents", run_agents_parallel_node)
    graph.add_node("scoring", aggregate_scores_node)
    graph.add_node("compare", compare_designs_node)
    graph.add_node("report", build_full_report_node)

    # Add edges
    if START:
        graph.add_edge(START, "preprocess")
    else:
        graph.set_entry_point("preprocess")

    graph.add_edge("preprocess", "rag")
    graph.add_edge("rag", "agents")
    graph.add_edge("agents", "scoring")
    graph.add_edge("scoring", "compare")
    graph.add_edge("compare", "report")
    graph.add_edge("report", END)

    # Compile graph
    compiled_graph = graph.compile()

    return compiled_graph


def aggregate_results_node(state):
    """
    Function 5.4: Aggregate all agent outputs into final report

    Args:
        state: Current analysis state

    Returns:
        dict: Updated state with final_report
    """
    print("📊 Aggregating results...")

    def safe_score(value):
        """Convert value to float if possible, else return 0."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    visual = state.get('visual_analysis', {})
    ux = state.get('ux_analysis', {})
    market = state.get('market_analysis', {})
    conversion = state.get('conversion_analysis', {})
    brand = state.get('brand_analysis', {})
    agent_errors = {}

    def _coerce_payload(name, payload):
        # If payload is not a dict or missing expected keys, flag it.
        if not isinstance(payload, dict):
            agent_errors[name] = "Invalid agent payload (non-dict)"
            return {"error": "Invalid agent payload"}
        if "error" in payload:
            agent_errors[name] = payload.get("error")
        return payload

    visual = _coerce_payload("visual", visual)
    ux = _coerce_payload("ux", ux)
    market = _coerce_payload("market", market)
    conversion = _coerce_payload("conversion", conversion)
    brand = _coerce_payload("brand", brand)

    # Calculate overall score (weighted average)
    visual_score = safe_score(visual.get('overall_score', 0))
    ux_score = safe_score(ux.get('overall_score', 0))
    market_score = safe_score(market.get('overall_score', 0))
    conversion_score = safe_score(conversion.get('overall_score', 0))
    brand_score = safe_score(brand.get('overall_score', 0))

    overall_score = (
        visual_score * 0.2 +
        ux_score * 0.2 +
        market_score * 0.2 +
        conversion_score * 0.2 +
        brand_score * 0.2
    )

    # Aggregate all recommendations
    all_recommendations = []

    # From visual analysis
    if 'color_analysis' in visual:
        print("📊 Aggregating results color_analysis ...")
        recs = visual['color_analysis'].get('recommendations', [])
        for rec in recs[:3]:  # Top 3
            all_recommendations.append({
                "source": "Visual - Color",
                "priority": "high",
                "recommendation": rec
            })

    if 'layout_analysis' in visual:
        print("📊 Aggregating results layout_analysis ...")
        recs = visual['layout_analysis'].get('recommendations', [])
        for rec in recs[:2]:
            all_recommendations.append({
                "source": "Visual - Layout",
                "priority": "medium",
                "recommendation": rec
            })

    if 'typography' in visual:
        print("📊 Aggregating results typography ...")
        recs = visual['typography'].get('recommendations', [])
        for rec in recs[:2]:
            all_recommendations.append({
                "source": "Visual - Typography",
                "priority": "medium",
                "recommendation": rec
            })

    # From UX analysis
    if 'usability' in ux:
        print("📊 Aggregating results usability ...")
        recs = ux['usability'].get('recommendations', [])
        for rec in recs[:3]:
            all_recommendations.append({
                "source": "UX - Usability",
                "priority": "high",
                "recommendation": rec
            })

    if 'accessibility' in ux:
        print("📊 Aggregating results accessibility ...")
        recs = ux['accessibility'].get('recommendations', [])
        for rec in recs[:2]:
            all_recommendations.append({
                "source": "UX - Accessibility",
                "priority": "critical",
                "recommendation": rec
            })

    # From market analysis
    if 'platform_optimization' in market:
        print("📊 Aggregating results platform_optimization ...")
        recs = market['platform_optimization'].get('recommendations', [])
        for rec in recs[:2]:
            all_recommendations.append({
                "source": "Market - Platform",
                "priority": "high",
                "recommendation": rec
            })

    if 'engagement_prediction' in market:
        print("📊 Aggregating results engagement_prediction ...")
        tips = market['engagement_prediction'].get('optimization_tips', [])
        for tip in tips[:3]:
            all_recommendations.append({
                "source": "Market - Engagement",
                "priority": "medium",
                "recommendation": tip
            })

    # From conversion analysis
    if 'cta' in conversion:
        recs = conversion['cta'].get('recommendations', [])
        for rec in recs[:3]:
            all_recommendations.append({
                "source": "Conversion - CTA",
                "priority": "high",
                "recommendation": rec
            })

    if 'copy' in conversion:
        recs = conversion['copy'].get('recommendations', [])
        for rec in recs[:2]:
            all_recommendations.append({
                "source": "Conversion - Copy",
                "priority": "medium",
                "recommendation": rec
            })

    if 'funnel_fit' in conversion:
        recs = conversion['funnel_fit'].get('recommendations', [])
        for rec in recs[:2]:
            all_recommendations.append({
                "source": "Conversion - Funnel",
                "priority": "medium",
                "recommendation": rec
            })

    # From brand analysis
    if 'logo_usage' in brand:
        recs = brand['logo_usage'].get('recommendations', [])
        for rec in recs[:2]:
            all_recommendations.append({
                "source": "Brand - Logo",
                "priority": "high",
                "recommendation": rec
            })

    if 'palette_alignment' in brand:
        recs = brand['palette_alignment'].get('recommendations', [])
        for rec in recs[:2]:
            all_recommendations.append({
                "source": "Brand - Palette",
                "priority": "medium",
                "recommendation": rec
            })

    if 'typography_alignment' in brand:
        recs = brand['typography_alignment'].get('recommendations', [])
        for rec in recs[:2]:
            all_recommendations.append({
                "source": "Brand - Typography",
                "priority": "medium",
                "recommendation": rec
            })

    if 'tone_voice' in brand:
        recs = brand['tone_voice'].get('recommendations', [])
        for rec in recs[:2]:
            all_recommendations.append({
                "source": "Brand - Tone",
                "priority": "medium",
                "recommendation": rec
            })

    # Limit to top 10 recommendations
    prioritized_recommendations = all_recommendations[:10]

    # Build final report
    state['final_report'] = {
        "overall_score": round(overall_score, 1),
        "agent_scores": {
            "visual": round(visual_score, 1),
            "ux": round(ux_score, 1),
            "market": round(market_score, 1),
            "conversion": round(conversion_score, 1),
            "brand": round(brand_score, 1)
        },
        "top_recommendations": prioritized_recommendations,
        "detailed_findings": {
            "visual": visual,
            "ux": ux,
            "market": market,
            "conversion": conversion,
            "brand": brand
        },
        "agent_errors": agent_errors,
        "metadata": state.get('image_metadata', {}),
        "platform": state.get('platform', 'Unknown'),
        "creative_type": state.get('creative_type', 'General'),
        "timestamp": datetime.now().isoformat(),
        "model_used": state.get('model_used', 'Unknown')
    }

    state['current_step'] = state.get('current_step', 0) + 1
    print("📊 Aggregating results completed!")
    return state


def _maybe_skip(agent_name, output_key, agent_fn):
    """
    Wrap agent to allow skipping when disabled by user selection.
    """
    def runner(state):
        enabled = state.get("enabled_agents", [])
        if enabled and agent_name not in enabled:
            state[output_key] = {"error": "skipped_by_user"}
            state['current_step'] = state.get('current_step', 0) + 1
            return state
        return agent_fn(state)
    return runner


def create_orchestration_graph(faiss_index, metadata):
    """
    Function 5.2: Build LangGraph workflow

    Args:
        faiss_index: FAISS index for RAG
        metadata: Pattern metadata

    Returns:
        Compiled LangGraph application
    """
    from components.agents import visual_analysis_agent, ux_critique_agent, market_research_agent, conversion_optimization_agent, brand_consistency_agent

    # Initialize graph
    workflow = StateGraph(AnalysisState)

    # Add nodes (agents)
    workflow.add_node(
        "visual_agent",
        _maybe_skip(
            "visual",
            "visual_analysis",
            lambda state: visual_analysis_agent(
                state, faiss_index, metadata, state.get("top_k", 3))
        )
    )
    workflow.add_node(
        "ux_agent",
        _maybe_skip(
            "ux",
            "ux_analysis",
            lambda state: ux_critique_agent(
                state, faiss_index, metadata, state.get("top_k", 3))
        )
    )
    workflow.add_node(
        "market_agent",
        _maybe_skip(
            "market",
            "market_analysis",
            lambda state: market_research_agent(
                state, faiss_index, metadata, state.get("top_k", 3))
        )
    )
    workflow.add_node(
        "conversion_agent",
        _maybe_skip(
            "conversion",
            "conversion_analysis",
            lambda state: conversion_optimization_agent(
                state, faiss_index, metadata, state.get("top_k", 3))
        )
    )
    workflow.add_node(
        "brand_agent",
        _maybe_skip(
            "brand",
            "brand_analysis",
            lambda state: brand_consistency_agent(
                state, faiss_index, metadata, state.get("top_k", 3))
        )
    )
    workflow.add_node("aggregator", aggregate_results_node)

    # Define edges (sequential flow)
    workflow.set_entry_point("visual_agent")
    workflow.add_edge("visual_agent", "ux_agent")
    workflow.add_edge("ux_agent", "market_agent")
    workflow.add_edge("market_agent", "conversion_agent")
    workflow.add_edge("conversion_agent", "brand_agent")
    workflow.add_edge("brand_agent", "aggregator")
    workflow.add_edge("aggregator", END)

    # Compile
    app = workflow.compile()

    print("✅ LangGraph workflow created")
    return app


def execute_analysis_workflow(graph, initial_state, progress_callback=None):
    """
    Function 5.3: Execute LangGraph workflow with progress tracking

    Args:
        graph: Compiled LangGraph application
        initial_state: Initial analysis state
        progress_callback: Function to call for progress updates

    Returns:
        dict: Final state after all agents complete
    """
    step_names = ["Visual Analysis", "UX Critique", "Market Research",
                  "Conversion Optimization", "Brand Consistency", "Aggregating Results"]
    total_steps = len(step_names)

    try:
        # Execute graph
        final_state = None
        current_step = 0

        for output in graph.stream(initial_state):
            # Update progress
            if progress_callback and current_step < len(step_names):
                progress_callback(
                    current_step + 1,
                    total_steps,
                    f"🔄 {step_names[current_step]}..."
                )

            # LangGraph stream yields dicts keyed by node name; ignore __end__ events
            if isinstance(output, dict):
                for key in ['aggregator', 'market_agent', 'ux_agent', 'visual_agent']:
                    if key in output:
                        final_state = output[key]
                        break

            current_step += 1

        return final_state or {}

    except Exception as e:
        print(f"❌ Error in workflow execution: {e}")
        return {
            "error": str(e),
            "final_report": {
                "overall_score": 0,
                "agent_scores": {"visual": 0, "ux": 0, "market": 0},
                "top_recommendations": [],
                "error_message": str(e)
            }
        }
