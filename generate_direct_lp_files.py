#!/usr/bin/env python3
"""
generate_direct_lp_files.py

Python script that generates LP files using satisfaction-based classification approach.
This script uses the same classification prompt as RCV but asks the LLM to identify 
which GDPR requirements (if any) a DPA segment SATISFIES.

The approach is:
1. For each segment, present ALL requirements and ask: "Which requirements does this segment SATISFY?"
2. Generate symbolic LP files with facts that indicate satisfaction/non-satisfaction
3. Use Deolingo to process the symbolic logic and determine final status
4. Use the same evaluation pipeline as RCV approach

This script generates .lp files that are processed by Deolingo solver.
"""

import os
import json
import argparse
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm
from ollama_client import OllamaClient


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Symbolic LP files for DPA segments using satisfaction-based classification"
    )
    parser.add_argument(
        "--requirements",
        type=str,
        required=True,
        help="Path to requirements JSON file"
    )
    parser.add_argument(
        "--dpa_segments",
        type=str,
        required=True,
        help="Path to DPA segments CSV file"
    )
    parser.add_argument(
        "--target_dpa",
        type=str,
        required=True,
        help="Target DPA name to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for LP files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:32b",
        help="Ollama model to use (default: qwen2.5:32b)"
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=0,
        help="Maximum number of segments to process (0 means all, default: 0)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def load_requirements(file_path: str) -> Dict:
    """Load requirements from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_dpa_segments(file_path: str, target_dpa: str, max_segments: int = 0) -> pd.DataFrame:
    """Load and filter DPA segments."""
    df = pd.read_csv(file_path)
    
    # Filter for target DPA
    df_filtered = df[df['DPA'] == target_dpa].copy()
    
    if df_filtered.empty:
        raise ValueError(f"No segments found for DPA: {target_dpa}")
    
    # Apply segment limit if specified
    if max_segments > 0:
        df_filtered = df_filtered.head(max_segments)
    
    # Reset index for consistent iteration
    df_filtered = df_filtered.reset_index(drop=True)
    
    return df_filtered


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


def classify_segment_satisfaction(segment_text: str, requirements: Dict, llm_client: OllamaClient, model: str, verbose: bool = False) -> Set[str]:
    """Classify which requirements (if any) a segment SATISFIES.
    
    Returns:
        Set of requirement IDs that the segment satisfies
    """
    
    # Build requirements list for system prompt
    req_list = []
    for req_id, req_info in requirements.items():
        req_list.append(f"{req_id}: {req_info['text']}")
    
    # --- SATISFACTION-BASED CLASSIFICATION PROMPT (BASED ON RCV) ---
    classification_system_prompt = f"""
# ROLE & GOAL
You are a legal expert specializing in GDPR compliance analysis. Your primary goal is to analyze a given DPA segment and identify which GDPR requirements (if any) it SATISFIES.

# INPUT DATA
1.  **AVAILABLE GDPR REQUIREMENTS**: A list of requirements, each with an ID and text.
    - {chr(10).join(req_list)}
2.  **DPA SEGMENT**: The text clause to be analyzed.

# CORE TASK
Analyze the DPA segment to determine which requirements it SATISFIES. A segment satisfies a requirement if it contains specific, actionable obligations that fulfill or implement that requirement.

**OUTPUT FORMAT**: Return a space-separated list of requirement IDs (e.g., "3 7 15") or "OTHER" if the segment doesn't satisfy any requirement. Do not output anything else, only a number or OTHER, no EXPLANATIONS.

---

# SATISFACTION METHODOLOGY

## Step 1: Initial Evaluation
Before classifying, you MUST apply these three tests. If a segment is merely descriptive, it satisfies nothing.
1.  **ACTIONABLE TEST**: Does this segment describe a SPECIFIC, CONCRETE action the processor MUST or MUST NOT perform?
2.  **OBLIGATION TEST**: Is this a binding contractual obligation that can be enforced?
3.  **DESCRIPTIVE TEST**: Is this merely describing, defining, or providing context without creating an obligation?

## Step 2: Satisfaction Analysis

### General Satisfaction Rules
- Focus EXCLUSIVELY on specific, actionable processor obligations that FULFILL requirements.
- Identify concrete data protection measures with clear duties that IMPLEMENT GDPR requirements.
- Recognize indirect obligations (e.g., processor actions that assist controller compliance).
- Consider alternative terminology (e.g., "customers" or "users" may mean "data subjects").
- Classify VIOLATIONS/NEGATIONS as NOT satisfying the requirement (e.g., "processor will NOT do X" does NOT satisfy the requirement to "do X").

### Context: Breach vs. Notification (R9 vs. R10)
- **9 (Assist Supervisory Authority)**: Satisfied if the processor helps the controller report to regulators (e.g., "supervisory authority", "regulatory body"). This is for EXTERNAL reporting to authorities.
- **10 (Assist Data Subjects)**: Satisfied if the processor helps the controller notify affected individuals (e.g., "data subject", "individual", "users/customers"). This is for EXTERNAL notification to people.
- **IGNORE Processor -> Controller Notices**: If the clause is ONLY about the processor notifying the CONTROLLER of a breach, this does NOT satisfy R9 or R10. Requirements 9 and 10 are strictly about assisting with notifications to external parties.

### Context: Security (R6 - Article 32)
- **STRICT CRITERIA**: Satisfies R6 ONLY if the segment describes:
    1.  SPECIFIC Article 32 measures: **pseudonymization, encryption, confidentiality, integrity, availability, resilience, restoration, or regular testing**.
    2.  EXPLICIT reference to "Article 32" or "security of processing" with concrete obligations.
- **EXPANDED KEYWORDS**: Also treat obligations for **availability, resilience, restoring access, log retention for security investigation, disaster-recovery, backup rotation, penetration testing, and business-continuity plans** as satisfying R6.
- **DO NOT SATISFY R6**:
    - General IT security (firewalls, network security).
    - Vague security statements ("shall comply with security policies").
    - Administrative security policies without implementation details.

### Context: Sub-Processors (R1 vs. R2 vs. R18)
- **1 (Authorization)**: Satisfied by prohibition on engaging sub-processors without *prior written consent*.
- **2 (Notification)**: Satisfied by obligation to *inform the controller* of intended additions or replacements.
- **18 (Liability/Oversight)**: Satisfied when the processor *remains fully liable* for the sub-processor, or has duties to *monitor, control, or audit* them. If you see these oversight verbs, this satisfies **18**, not 1 or 2.

### Context: Contract Breach vs. Data Breach
- **HARD FILTER**: Treat any mention of "breach" preceded by "contract", "agreement", or "service" as a contract breach. This does NOT satisfy any data protection requirement.

## Exclusion Criteria (Content that Satisfies Nothing)
- Administrative headers, titles, appendices, section numbers.
- Definitions or explanatory text without actionable obligations.
- General compliance statements (e.g., "shall comply with applicable laws").
- Descriptive statements about processing scope or data categories.
- Short, meaningless phrases (< 10 words) or segments starting with "Article", "Section", etc.
- Vague references to security measures being described elsewhere (e.g., "Security measures are described in Appendix B.").

# EXAMPLES

**EXAMPLES (REQUIREMENTS SATISFIED):**

Input: "The processor shall not subcontract any of its processing operations performed on behalf of the controller under the Clauses without the prior written consent of the controller."
Output: 1

Input: "processor shall notify controller with at least ten (10) days' prior notice before authorizing any new Sub-Processors to access controller's Personal Data;"
Output: 2

Input: "processor will process controller Data only in accordance with Documented Instructions."
Output: 3

Input: "In such cases, the processor shall inform the controller of the legal requirement before processing, unless that law prohibits such information on important grounds of public interest."
Output: 4

Input: "processor imposes appropriate contractual obligations upon its personnel, including relevant obligations regarding confidentiality, data protection and data security."
Output: 5

Input: "The processor implements multi-factor authentication for all system access."
Output: 6

Input: "All customer data is encrypted both at rest and in transit using industry-standard encryption algorithms."
Output: 6

Input: "The processor maintains backup systems to ensure data availability and implements disaster recovery procedures."
Output: 6

Input: "In the event any such request is made directly to processor, processor shall notify controller in writing of such request promptly upon receipt thereof."
Output: 7

Input: "The processor may add to, amend, or replace the specific security measures for security reasons and shall notify the controller in writing where it has done so."
Output: 8

Input: "If the Controller has an obligation to notify or report in the event of a security incident, the Processor is obliged to support the Controller at the Controllers expense."
Output: 10

Input: "The processor shall assist the controller in communicating personal data breaches to affected individuals."
Output: 10

Input: "Processor shall provide reasonable assistance to Controller with any data protection impact assessments."
Output: 11

Input: "processor will delete controller Data when requested by controller by using the Service controls provided for this purpose by processor."
Output: 13

Input: "processor shall inform controller if, in the opinion of processor, an instruction infringes Applicable Data Protection Law."
Output: 14

Input: "processor will make available to controller all information necessary to demonstrate compliance with the obligations of Data Processors laid down in Article 28 of GDPR."
Output: 15

Input: "processor shall remain fully liable to controller for the performance of the Sub-Processor's obligations."
Output: 18

Input: "processor has implemented and will maintain appropriate technical and organizational security measures for the Processing of Personal Data."
Output: 19

**EXAMPLES (VIOLATIONS - DO NOT SATISFY):**

Input: "The processor shall engage sub-processors without any authorization from the controller."
Output: OTHER

Input: "The processor shall not inform the controller of any intended changes concerning sub-processors."
Output: OTHER

Input: "The processor may process personal data without documented instructions from the controller."
Output: OTHER

Input: "Personnel are not required to maintain confidentiality of personal data."
Output: OTHER

Input: "The processor shall not implement security measures for personal data processing."
Output: OTHER

**EXAMPLES (NON-RELEVANT - SATISFY NOTHING):**

Input: "This DPA applies when controller Data is processed by processor."
Output: OTHER

Input: "The subject matter of the data processing under this DPA is controller Data."
Output: OTHER

Input: "processor Agreement"
Output: OTHER

Input: "Appendix A"
Output: OTHER

Input: "The European Parliament and the Council's Directive 95/46/EF of 24 October 1995 on the protection of individuals with regard to the processing of personal data"
Output: OTHER

Input: "The processor shall comply with applicable data protection laws"
Output: OTHER

Input: "Security measures are described in Appendix B."
Output: OTHER

Input: "The processor's liability for breach of contract shall be limited to annual fees"
Output: OTHER
"""
    
    user_prompt = segment_text
    
    try:
        response = llm_client.generate(user_prompt, model_name=model, system_prompt=classification_system_prompt)
        # Filter out thinking sections
        response = filter_think_sections(response)
        response = response.strip()
        
        if verbose:
            print(f"Classification response: {response}")
        
        # Parse response - extract only valid requirement IDs
        if response == "OTHER":
            return set()
        
        # Parse space-separated requirement IDs, handling various response formats
        satisfied_reqs = set()
        
        # Split by various delimiters and extract numbers
        import re
        # Look for patterns like "1", "2", "3", etc. or "R1", "R2", etc.
        id_pattern = r'\\b(?:R?([1-9]|1[0-9]|2[0-9]))\\b'
        matches = re.findall(id_pattern, response)
        
        for match in matches:
            req_id = match
            if req_id in requirements:
                satisfied_reqs.add(req_id)
            elif verbose:
                print(f"Warning: Invalid requirement ID '{req_id}' found in response")
        
        # If no valid IDs found with regex, try simple splitting
        if not satisfied_reqs:
            parts = response.split()
            for part in parts:
                # Clean up the part (remove punctuation, etc.)
                clean_part = re.sub(r'[^0-9]', '', part)
                if clean_part and clean_part in requirements:
                    satisfied_reqs.add(clean_part)
                elif verbose and clean_part:
                    print(f"Warning: Invalid requirement ID '{clean_part}' in response")
        
        return satisfied_reqs
            
    except Exception as e:
        print(f"Error in classification: {e}")
        return set()


def extract_body_atoms(symbolic_rule):
    """Extract atoms from the body of an ASP rule."""
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


def generate_symbolic_lp_file(segment_text: str, req_text: str, req_symbolic: str, req_predicates: List[str], is_satisfied: bool) -> str:
    """Generate LP file content with symbolic logic for Deolingo processing."""
    
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
    
    # Add facts based on satisfaction classification
    lp_content += "% 2. Facts based on satisfaction classification\n"
    if is_satisfied:
        # If satisfied, add the main predicate and role(processor)
        # Extract the main predicate from the symbolic rule
        if "&obligatory{" in req_symbolic:
            predicate = req_symbolic.split("&obligatory{")[1].split("}")[0]
            # Handle negative predicates properly
            if predicate.startswith('-'):
                # For negative predicates, satisfaction means the predicate is NOT true
                # So we don't add the positive predicate
                lp_content += f"% Negative predicate {predicate} - satisfied by absence\n"
            else:
                lp_content += f"{predicate}.\n"
        elif "&forbidden{" in req_symbolic:
            predicate = req_symbolic.split("&forbidden{")[1].split("}")[0]
            # For forbidden predicates, we don't add the predicate (absence means compliance)
            if predicate.startswith('-'):
                # For negative forbidden predicates, add the positive predicate
                positive_predicate = predicate[1:]  # Remove the '-'
                lp_content += f"{positive_predicate}.\n"
            else:
                lp_content += f"% Forbidden predicate {predicate} - satisfied by absence\n"
        elif "&permitted{" in req_symbolic:
            predicate = req_symbolic.split("&permitted{")[1].split("}")[0]
            if predicate.startswith('-'):
                # For negative permitted predicates, don't add the positive predicate
                lp_content += f"% Negative permitted predicate {predicate} - satisfied by absence\n"
            else:
                lp_content += f"{predicate}.\n"
        
        # Always add role(processor) for satisfied requirements
        lp_content += "role(processor).\n"
        
        # Add any additional body atoms that might be needed
        for atom in body_atoms:
            if atom not in ["role(processor)"]:  # Don't duplicate role(processor)
                lp_content += f"{atom}.\n"
    else:
        # If not satisfied, only add role(processor) but not the main predicate
        lp_content += "role(processor).\n"
        lp_content += "% Segment does not satisfy this requirement\n"
    
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


def process_dpa_segments(segments_df: pd.DataFrame, requirements: Dict, llm_client: OllamaClient, 
                        model: str, output_dir: str, verbose: bool = False) -> None:
    """Process all DPA segments using satisfaction-based classification and generate symbolic LP files."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each segment
    for idx, row in tqdm(segments_df.iterrows(), total=len(segments_df), desc="Processing segments"):
        segment_id = row["ID"]
        segment_text = row["Sentence"]
        
        if verbose:
            print(f"\nProcessing segment {segment_id}: {segment_text[:100]}...")
        
        # Classify segment to find which requirements it satisfies
        satisfied_req_ids = classify_segment_satisfaction(
            segment_text, requirements, llm_client, model, verbose
        )
        
        if verbose:
            if satisfied_req_ids:
                print(f"Satisfies requirements: {', '.join(satisfied_req_ids)}")
            else:
                print("Satisfies no requirements (OTHER)")
        
        # Generate LP files for all requirements (to maintain compatibility with evaluation)
        for req_id, requirement_info in requirements.items():
            req_text = requirement_info["text"]
            req_symbolic = requirement_info["symbolic"]
            req_predicates = requirement_info["atoms"]
            
            # Create requirement directory
            req_dir = os.path.join(output_dir, f"req_{req_id}")
            os.makedirs(req_dir, exist_ok=True)
            
            # Check if this requirement is satisfied by the segment
            is_satisfied = req_id in satisfied_req_ids
            lp_content = generate_symbolic_lp_file(
                segment_text, req_text, req_symbolic, req_predicates, is_satisfied
            )
            
            # Save LP file
            lp_file_path = os.path.join(req_dir, f"segment_{segment_id}.lp")
            with open(lp_file_path, 'w') as f:
                f.write(lp_content)
        
        if verbose:
            print(f"Generated symbolic LP files for segment {segment_id}")


def main():
    """Main function."""
    args = parse_arguments()
    
    print("========== Symbolic LP File Generator (Satisfaction-Based Classification) ==========")
    print(f"Target DPA: {args.target_dpa}")
    print(f"Model: {args.model}")
    print(f"Output Directory: {args.output}")
    print("====================================================================================")
    
    # Initialize Ollama client
    print("Initializing Ollama client...")
    llm_client = OllamaClient()
    
    if not llm_client.check_health():
        print("Error: Ollama server is not running. Please start it first.")
        return 1
    
    # Load requirements
    print(f"Loading requirements from: {args.requirements}")
    requirements = load_requirements(args.requirements)
    print(f"Loaded {len(requirements)} requirements")
    
    # Load DPA segments
    print(f"Loading DPA segments from: {args.dpa_segments}")
    segments_df = load_dpa_segments(args.dpa_segments, args.target_dpa, args.max_segments)
    print(f"Loaded {len(segments_df)} segments for DPA: {args.target_dpa}")
    
    # Process segments and generate LP files
    print("Processing segments with satisfaction-based classification...")
    process_dpa_segments(
        segments_df, requirements, llm_client, args.model, 
        args.output, args.verbose
    )
    
    print(f"\nSymbolic LP file generation completed!")
    print(f"Generated {len(segments_df)} symbolic LP files in: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 
