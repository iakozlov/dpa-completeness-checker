# generate_lp_files.py
import os
import json
import argparse
import pandas as pd
import re
from tqdm import tqdm
from ollama_client import OllamaClient

def filter_think_sections(text):
    """
    Remove <think> sections from model responses.
    
    Args:
        text (str): The raw model response
        
    Returns:
        str: The filtered text with <think> sections removed
    """
    # Use regex to remove everything between <think> and </think> tags (case insensitive, multiline)
    filtered = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any extra whitespace that might be left
    filtered = re.sub(r'\n\s*\n', '\n', filtered.strip())
    
    return filtered

def main():
    parser = argparse.ArgumentParser(description="Generate LP files for DPA segments")
    parser.add_argument("--requirements", type=str, default="data/requirements/requirements_deontic_manual.json",
                        help="Path to requirements deontic JSON file")
    parser.add_argument("--dpa", type=str, default="data/test_set.csv",
                        help="Path to DPA segments CSV file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use (gpt-4o-mini, llama2-70b, mistral-7b, etc.)")
    parser.add_argument("--output", type=str, default="results/lp_files",
                        help="Output directory for LP files")
    parser.add_argument("--target_dpa", type=str, default="Online 126",
                        help="Target DPA to process (default: Online 124)")
    parser.add_argument("--req_ids", type=str, default="all",
                        help="Comma-separated list of requirement IDs to process, or 'all' (default: all)")
    parser.add_argument("--max_segments", type=int, default=0,
                        help="Maximum number of segments to process (0 means all, default: 0)")
    parser.add_argument("--debug_req_id", type=str,
                        help="Debug mode: Process only this specific requirement ID")
    parser.add_argument("--debug_segment_id", type=str, 
                        help="Debug mode: Process only this specific segment ID")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output for debugging")

    parser.add_argument("--requirement_prompts", type=str, default="requirement_prompts.json",
                        help="Path to requirement-specific prompts JSON file")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load requirements
    print(f"Loading requirements from: {args.requirements}")
    with open(args.requirements, 'r') as f:
        all_requirements = json.load(f)
    
    # Load requirement-specific prompts if available
    requirement_prompts = {}
    if os.path.exists(args.requirement_prompts):
        print(f"Loading requirement-specific prompts from: {args.requirement_prompts}")
        with open(args.requirement_prompts, 'r') as f:
            requirement_prompts = json.load(f)
        print(f"Loaded prompts for {len(requirement_prompts)} requirements")
    else:
        print(f"Warning: Requirement prompts file not found at {args.requirement_prompts}")
        print("Using generic system prompt for all requirements")
    
    # Filter requirements by ID if specified
    if args.req_ids.lower() != "all":
        req_ids = [id.strip() for id in args.req_ids.split(",")]
        requirements = {id: all_requirements[id] for id in req_ids if id in all_requirements}
        if not requirements:
            print(f"Error: No valid requirement IDs found. Available IDs: {', '.join(all_requirements.keys())}")
            return
        print(f"Processing {len(requirements)} requirements with IDs: {', '.join(requirements.keys())}")
    else:
        requirements = all_requirements
        print(f"Processing all {len(requirements)} requirements")
    
    # Load DPA segments
    print(f"Loading DPA segments from: {args.dpa}")
    df = pd.read_csv(args.dpa)
    
    # Filter for the target DPA only
     # Filter for the target DPA only
    target_dpa = args.target_dpa
    df_filtered = df[df['DPA'] == target_dpa]
    
    if df_filtered.empty:
        print(f"Error: DPA '{target_dpa}' not found in the dataset.")
        return
    
    # Apply segment limit if specified
    if args.max_segments > 0:
        df_filtered = df_filtered.head(args.max_segments)
        print(f"Processing first {len(df_filtered)} segments for DPA: {target_dpa}")
    else:
        print(f"Processing all {len(df_filtered)} segments for DPA: {target_dpa}")
    
    # Create directory for this DPA
    dpa_dir = os.path.join(args.output, f"dpa_{target_dpa.replace(' ', '_')}")
    os.makedirs(dpa_dir, exist_ok=True)

        # Initialize LLM (Ollama only)
    print(f"Initializing Ollama client with model: {args.model}")
    llm_model = OllamaClient()
    if not llm_model.check_health():
        print("Error: Ollama server is not running. Please start it first.")
        return
    print("Ollama client initialized successfully")
    
    # Process each requirement
    for req_id, req_info in tqdm(requirements.items(), desc="Processing requirements"):
        # Skip if debug_req_id is specified and this isn't it
        if args.debug_req_id and req_id != args.debug_req_id:
            continue
            
        req_text = req_info["text"]
        req_symbolic = req_info["symbolic"]
        
        # Create directory for this requirement
        req_dir = os.path.join(dpa_dir, f"req_{req_id}")
        os.makedirs(req_dir, exist_ok=True)
        
        # Extract predicates from the requirement
        req_predicates = extract_predicates(req_info)
        
        # Process each segment
        for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"Processing segments for requirement {req_id}"):
            segment_id = row["ID"]
            
            # Skip if debug_segment_id is specified and this isn't it
            if args.debug_segment_id and str(segment_id) != args.debug_segment_id:
                continue
                
            segment_text = row["Sentence"]
            
            if args.verbose:
                print(f"\nProcessing segment {segment_id}:")
                print(f"Text: {segment_text}")
            
            # Generate LP file for this segment
            lp_file_path = os.path.join(req_dir, f"segment_{segment_id}.lp")
            
            # Extract facts from DPA segment
            facts = extract_facts_from_dpa(segment_text, req_text, req_symbolic, req_predicates, llm_model, args.model, req_id, requirement_prompts)
            
            # Generate LP file content
            lp_content = generate_lp_file(req_symbolic, facts, req_predicates, req_text, segment_text)
            
            if args.verbose:
                print(f"Generated LP content:\n{lp_content}")
            
            # Write to file
            with open(lp_file_path, 'w') as f:
                f.write(lp_content)
    
    print("LP file generation complete!")

def extract_predicates(req_info):
    """Extract predicates from the requirement's atoms field."""
    # Return the atoms directly from the requirement info
    return req_info.get("atoms", [])

def extract_facts_from_dpa(segment_text, req_text, req_symbolic, req_predicates, llm_model, model_name="qwen2.5:7b", req_id="", requirement_prompts={}):
    """Extract facts from a DPA segment using the LLM with enhanced prompt.
    
    Args:
        segment_text: The text of the DPA segment
        req_text: The text of the requirement
        req_symbolic: The symbolic representation of the requirement
        req_predicates: List of predicates from the requirement
        llm_model: The Ollama client to use for extraction
        model_name: Name of the Ollama model to use
        req_id: ID of the current requirement
        requirement_prompts: Dictionary of requirement-specific prompts
    Returns:
        Dictionary mapping predicates to their truth values
    """
    # Use the enhanced fact extraction prompt from RCV approach
    fact_extraction_system_prompt = """
# ROLE & GOAL
You are a legal text analysis expert. Your task is to analyze a DPA clause and extract symbolic facts based on a specific GDPR requirement.

# INPUT DATA
1.  **REQUIREMENT**: The text of a GDPR requirement.
2.  **SYMBOLIC**: The requirement's formal logic representation.
3.  **PREDICATES**: A list of all possible symbolic atoms for this requirement.
4.  **CLAUSE**: The DPA text segment to analyze.

# CORE TASK
Analyze the **CLAUSE** to see if it contains specific, actionable obligations that semantically match the **REQUIREMENT**. Extract the corresponding predicates.

# OUTPUT REQUIREMENTS
- Your output MUST be **ONLY** a semicolon-separated list of predicates (e.g., `predicate1;predicate2`).
- To indicate a violation or negation of an obligation, prefix the predicate with a hyphen (`-predicate3`).
- If no relevant facts are found, you MUST return the exact string `NO_FACTS`.
- DO NOT output any explanations, apologies, or introductory text.

# EXTRACTION PRINCIPLES & RULES

## Primary Rule: Obligation vs. Description
- Extract a fact ONLY if the **CLAUSE** describes a **specific, actionable processor obligation**.
- DO NOT extract facts for merely descriptive, administrative, or definitional text.
- The `role(processor)` predicate should only be extracted if a real obligation is present.

## Handling Violations / Negations
- If the CLAUSE describes an action that violates the REQUIREMENT, extract the relevant predicate.
  - REQ: "shall do X", CLAUSE: "will NOT do X" -> Extract `-do_x`.
  - REQ: "shall NOT do X", CLAUSE: "will do X" -> Extract `do_x`. (The logic program handles the violation).

## Context-Specific Rules
- **Breach Notifications (R9/R10)**:
    - `assist_notification_supervisory_authority` (R9): Only if the clause mentions helping the controller report to a **regulator/authority**.
    - `assist_communication_data_subject` (R10): Only if the clause mentions helping the controller notify **affected individuals**.
- **Security (R6)**:
    - `ensure_security_of_processing`: ONLY for specific GDPR Article 32 measures (encryption, pseudonymization, resilience, restoration, testing, etc.).
    - DO NOT extract for general IT security (firewalls) or vague compliance statements.
- **Sub-Processors (R1/R2)**:
    - `engage_sub_processor` (R1): For obligations related to the act of engaging/appointing.
    - `inform_controller_changes` (R2): For obligations related to notifying about changes.
- **Confidentiality (R5)**:
    - `ensure_confidentiality_commitment`: Only for specific, binding confidentiality duties imposed on personnel.

## General Guidance
- **Synonyms**: Recognize synonyms (e.g., "vendor" -> "processor"; "inform" -> "notify").
- **Complex Sentences**: Focus on the primary processor obligation(s) in long sentences.

# EXAMPLES

- **Example 1 (Precise Match):**
  REQUIREMENT: The processor shall ensure that persons authorized to process personal data have committed themselves to confidentiality.
  SYMBOLIC: &obligatory{ensure_confidentiality_commitment} :- role(processor).
  PREDICATES: ensure_confidentiality_commitment; role(processor)
  CLAUSE: The Processor shall ensure that every employee authorized to process Customer Personal Data is subject to a contractual duty of confidentiality.
  **Expected output:** ensure_confidentiality_commitment; role(processor)

- **Example 2 (Violation Detection):**
  REQUIREMENT: The processor shall take all measures required pursuant to Article 32 to ensure the security of processing.
  SYMBOLIC: &obligatory{ensure_security_of_processing} :- role(processor).
  PREDICATES: ensure_security_of_processing; role(processor)
  CLAUSE: The processor will not implement any security measures for customer data.
  **Expected output:** -ensure_security_of_processing; role(processor)

- **Example 3 (Administrative - No Facts):**
  REQUIREMENT: The processor shall process personal data only on documented instructions from the controller.
  SYMBOLIC: &obligatory{process_data_on_documented_instructions} :- role(processor).
  PREDICATES: process_data_on_documented_instructions; role(processor)
  CLAUSE: This Data Processing Addendum ("DPA") supplements the processor controller Agreement available at...
  **Expected output:** NO_FACTS

- **Example 4 (Article 32 Security - Specific):**
  REQUIREMENT: The processor shall take all measures required pursuant to Article 32 to ensure the security of processing.
  SYMBOLIC: &obligatory{ensure_security_of_processing} :- role(processor).
  PREDICATES: ensure_security_of_processing; role(processor)
  CLAUSE: All personal data is encrypted using AES-256 encryption both at rest and in transit.
  **Expected output:** ensure_security_of_processing; role(processor)

- **Example 5 (General Security - Role Only):**
  REQUIREMENT: The processor shall take all measures required pursuant to Article 32 to ensure the security of processing.
  SYMBOLIC: &obligatory{ensure_security_of_processing} :- role(processor).
  PREDICATES: ensure_security_of_processing; role(processor)
  CLAUSE: The processor implements network firewalls and intrusion detection systems for general IT security.
  **Expected output:** role(processor)

- **Example 6 (Breach Notification to Authority):**
  REQUIREMENT: The processor shall assist the controller in notifying a personal-data breach to the supervisory authority.
  SYMBOLIC: &obligatory{assist_notification_supervisory_authority} :- role(processor), data_breach.
  PREDICATES: role(processor); data_breach; assist_notification_supervisory_authority
  CLAUSE: The processor shall promptly notify the controller who will then report the incident to the relevant data protection authority.
  **Expected output:** assist_notification_supervisory_authority; role(processor)

- **Example 7 (Breach Notification to Subjects):**
  REQUIREMENT: The processor shall assist the controller in communicating a personal-data breach to the data subject.
  SYMBOLIC: &obligatory{assist_communication_data_subject} :- role(processor), data_breach.
  PREDICATES: role(processor); data_breach; assist_communication_data_subject
  CLAUSE: Processor will help controller notify affected individuals of any security incidents involving their personal data.
  **Expected output:** assist_communication_data_subject; role(processor)

- **Example 8 (Breach Definition - No Facts):**
  REQUIREMENT: The processor shall assist the controller in notifying a personal-data breach to the supervisory authority.
  SYMBOLIC: &obligatory{assist_notification_supervisory_authority} :- role(processor), data_breach.
  PREDICATES: role(processor); data_breach; assist_notification_supervisory_authority
  CLAUSE: A personal data breach means a breach of security leading to the accidental or unlawful destruction, loss, alteration, unauthorized disclosure.
  **Expected output:** NO_FACTS
"""

    user_prompt = f"""REQUIREMENT: {req_text}
SYMBOLIC: {req_symbolic}
PREDICATES: {'; '.join(req_predicates)}
CLAUSE: {segment_text}"""
    
    response = llm_model.generate(user_prompt, model_name=model_name, system_prompt=fact_extraction_system_prompt)
    
    # Filter out <think> sections from reasoning models like qwen3
    response = filter_think_sections(response)
    
    # Parse the response
    if response.strip() == "NO_FACTS":
        return {}
    facts = {}
    for pred in response.strip().split(';'):
        pred = pred.strip()
        if pred and pred != "NO_FACTS" and pred != "NO_FACTS.": 
            if pred.startswith('-'):
                facts[pred[1:]] = False
            else:
                facts[pred] = True
    
    # CRITICAL CHECK: If role(processor) is not among the extracted facts, 
    # treat this segment as NO_FACTS since GDPR requirements are about processor obligations.
    # This ensures we only consider segments that establish processor roles/obligations.
    # Segments that are purely definitional, administrative, or about other parties
    # should result in NO_FACTS (empty dictionary).
    if "role(processor)" not in facts:
        return {}
    
    return facts

def extract_body_atoms(symbolic_rule):
    """Extract atoms from the body of an ASP rule.
    
    Args:
        symbolic_rule (str): ASP rule in the format "head :- body."
        
    Returns:
        List[str]: List of atoms in the body
    """
    if ":-" not in symbolic_rule:
        return []
    
    # Split on :- to get the body part
    parts = symbolic_rule.split(":-")
    if len(parts) < 2:
        return []
    
    body = parts[1].strip()
    
    # Remove the trailing period
    if body.endswith('.'):
        body = body[:-1]
    
    # Split by comma to get individual atoms
    atoms = []
    for atom in body.split(','):
        atom = atom.strip()
        
        # Remove 'not ' prefix if present
        if atom.startswith('not '):
            atom = atom[4:].strip()
        
        # Add atom if it's not empty
        if atom:
            atoms.append(atom)
    
    return atoms

def generate_lp_file(req_symbolic, facts, req_predicates, req_text, segment_text):
    """Generate the content of an LP file."""
    # Start with the requirement's symbolic representation
    lp_content = f"% Requirement Text:\n% {req_text}\n%\n"
    lp_content += f"% DPA Segment:\n% {segment_text}\n%\n"
    
    # Extract body atoms from the symbolic rule
    body_atoms = extract_body_atoms(req_symbolic)
    
    # Add external declarations only for body atoms
    if body_atoms:
        lp_content += "% External declarations for rule body predicates\n"
        for atom in body_atoms:
            lp_content += f"#external {atom}.\n"
        lp_content += "\n"
    
    # Add the requirement's symbolic representation (normative layer)
    lp_content += "% 1. Normative layer\n"
    lp_content += f"{req_symbolic}\n\n"
    
    # Add facts
    lp_content += "% 2. Facts extracted from DPA segment\n"
    if facts:
        for pred, value in facts.items():
            if value:
                lp_content += f"{pred}.\n"
            else:
                lp_content += f"-{pred}.\n"
    else:
        lp_content += "% No semantically relevant facts found in this segment\n"
    
    lp_content += "\n"
    
    # Add status mapping - determine the deontic operator from the symbolic rule
    lp_content += "% 3. Map Deolingo's internal status atoms to our labels\n"
    
    # Extract the deontic operator and predicate from the symbolic rule
    if "&obligatory{" in req_symbolic:
        # Extract predicate from &obligatory{predicate}
        predicate = req_symbolic.split("&obligatory{")[1].split("}")[0]
        lp_content += f"status(satisfied)     :- &fulfilled_obligation{{{predicate}}}.\n"
        lp_content += f"status(violated)      :- &violated_obligation{{{predicate}}}.\n"
        lp_content += f"status(not_mentioned) :- &undetermined_obligation{{{predicate}}}.\n"
    elif "&forbidden{" in req_symbolic:
        # Extract predicate from &forbidden{predicate}
        predicate = req_symbolic.split("&forbidden{")[1].split("}")[0]
        lp_content += f"status(satisfied)     :- &fulfilled_prohibition{{{predicate}}}.\n"
        lp_content += f"status(violated)      :- &violated_prohibition{{{predicate}}}.\n"
        lp_content += f"status(not_mentioned) :- &undetermined_prohibition{{{predicate}}}.\n"
    elif "&permitted{" in req_symbolic:
        # Extract predicate from &permitted{predicate}
        predicate = req_symbolic.split("&permitted{")[1].split("}")[0]
        lp_content += f"status(satisfied)     :- &fulfilled_permission{{{predicate}}}.\n"
        lp_content += f"status(violated)      :- &violated_permission{{{predicate}}}.\n"
        lp_content += f"status(not_mentioned) :- &undetermined_permission{{{predicate}}}.\n"
    else:
        # Fallback for unknown deontic operators
        lp_content += "% Warning: Unknown deontic operator in symbolic rule\n"
        lp_content += "status(not_mentioned) :- true.\n"
    
    lp_content += "\n#show status/1.\n"
    
    return lp_content

if __name__ == "__main__":
    main()
