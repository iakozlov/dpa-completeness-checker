#!/usr/bin/env python3
"""
generate_rcv_lp_files.py

Python script that generates LP files using the RCV (Requirement Classification and Verification) approach.
This script follows the same pattern as generate_lp_files.py but implements the two-step RCV logic:
1. Classification Step: Determine which single GDPR requirement (if any) a segment is relevant to
2. Verification Step: Extract symbolic facts specific to that requirement

This script only generates .lp files - the solver is called separately by the shell script.
"""

import os
import json
import argparse
import pandas as pd
import re
from typing import Dict, List, Optional
from tqdm import tqdm
from ollama_client import OllamaClient


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate RCV LP files for DPA segments"
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
        default="llama3.3:70b",
        help="Ollama model to use (default: llama3.3:70b)"
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


def classify_segment(segment_text: str, requirements: Dict, llm_client: OllamaClient, model: str, verbose: bool = False) -> str:
    """Classify which requirement (if any) a segment is relevant to."""
    
    # Build requirements list for system prompt
    req_list = []
    for req_id, req_info in requirements.items():
        req_list.append(f"{req_id}: {req_info['text']}")
    
    # --- REFACTORED AND STRUCTURED PROMPT (FULL VERSION) ---
    classification_system_prompt = f"""
# ROLE & GOAL
You are a legal expert specializing in GDPR compliance analysis. Your primary goal is to classify a given DPA segment by identifying the single, most relevant GDPR requirement it pertains to.

# INPUT DATA
1.  **AVAILABLE GDPR REQUIREMENTS**: A list of requirements, each with an ID and text.
    - {chr(10).join(req_list)}
2.  **DPA SEGMENT**: The text clause to be analyzed.

# CORE TASK
Classify the DPA segment by returning ONLY the requirement ID (e.g., "3") or "NONE" if it is not relevant to any requirement.
Output only a number or NONE, DO NOT output any explanation.
---

# CLASSIFICATION METHODOLOGY

## Step 1: Initial Evaluation
Before classifying, you MUST apply these three tests. If a segment is merely descriptive, it is "NONE".
1.  **ACTIONABLE TEST**: Does this segment describe a SPECIFIC, CONCRETE action the processor MUST or MUST NOT perform?
2.  **OBLIGATION TEST**: Is this a binding contractual obligation that can be enforced?
3.  **DESCRIPTIVE TEST**: Is this merely describing, defining, or providing context without creating an obligation?

## Step 2: Classification and Disambiguation Rules

### General Classification Rules
- Focus EXCLUSIVELY on specific, actionable processor obligations.
- Identify concrete data protection measures with clear duties.
- Recognize indirect obligations (e.g., processor actions that assist controller compliance).
- Consider alternative terminology (e.g., "customers" or "users" may mean "data subjects").
- Classify VIOLATIONS/NEGATIONS as relevant to the requirement they violate (e.g., "processor will NOT do X" is relevant to the requirement to "do X").

### Context: Breach vs. Notification (R9 vs. R10)
- **R9 (Assist Supervisory Authority)**: Relevant if the processor helps the controller report to regulators (e.g., "supervisory authority", "regulatory body"). This is for EXTERNAL reporting to authorities.
- **R10 (Assist Data Subjects)**: Relevant if the processor helps the controller notify affected individuals (e.g., "data subject", "individual", "users/customers"). This is for EXTERNAL notification to people.
- **IGNORE Processor -> Controller Notices**: If the clause is ONLY about the processor notifying the CONTROLLER of a breach, classify as **NONE**. R9 and R10 are strictly about assisting with notifications to external parties.

### Context: Security (R6 - Article 32)
- **STRICT CRITERIA**: Classify as R6 ONLY if the segment describes:
    1. **CORE Article 32 measures ONLY**: 
       - Pseudonymization of personal data
       - Encryption of personal data
       - Ensuring confidentiality, integrity, availability, or resilience of processing systems
       - Ability to restore availability and access to personal data after incidents
       - Regular testing/evaluation specifically of these Article 32 measures
    2. **EXPLICIT reference to "Article 32" or "security of processing"**
- **DO NOT CLASSIFY AS R6**:
    - General IT security (firewalls, network security, patches)
    - Penetration testing (unless explicitly for Article 32 compliance)
    - Audit logging or access logs
    - General security policies or compliance statements
    - Security measures not specifically listed in Article 32

### Context: Sub-Processors (R1 vs. R2 vs. R18)
- **R1 (Authorization)**: Prohibition on engaging sub-processors without *prior written consent*.
- **R2 (Notification)**: Obligation to *inform the controller* of intended additions or replacements.
- **R18 (Liability/Oversight)**: The processor *remains fully liable* for the sub-processor, or has duties to *monitor, control, or audit* them. If you see these oversight verbs, choose **18**, not 1 or 2.

### Context: Contract Breach vs. Data Breach
- **HARD FILTER**: Treat any mention of "breach" preceded by "contract", "agreement", or "service" as a contract breach. Classify as **NONE**.

## Exclusion Criteria (Content to Classify as "NONE")
- Administrative headers, titles, appendices, section numbers.
- Definitions or explanatory text without actionable obligations.
- General compliance statements (e.g., "shall comply with applicable laws").
- Descriptive statements about processing scope or data categories.
- Short, meaningless phrases (< 10 words) or segments starting with "Article", "Section", etc.
- Vague references to security measures being described elsewhere (e.g., "Security measures are described in Appendix B.").

# EXAMPLES

**EXAMPLES (POSITIVE REQUIREMENTS):**

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

Input: "Guests and visitors to processor buildings must register their names at reception and must be accompanied by authorized processor personnel."
Output: 6

Input: "The processor implements multi-factor authentication for all system access."
Output: 6

Input: "Regular security audits and penetration testing shall be conducted by the processor."
Output: 6

Input: "All customer data is encrypted both at rest and in transit using industry-standard encryption algorithms."
Output: 6

Input: "The processor maintains backup systems to ensure data availability and implements disaster recovery procedures."
Output: 6

Input: "The security of the channel used must correspond to the privacy risk involved."
Output: 6

Input: "To maintain up-to-date compliance with the NHS Data Security and Protection Toolkit."
Output: 6

Input: "The processor implements pseudonymization and encryption of personal data to ensure security of processing."
Output: 6

Input: "The processor ensures ongoing confidentiality, integrity, availability and resilience of processing systems and services."
Output: 6

Input: "The processor maintains the ability to restore availability and access to personal data in a timely manner in the event of a physical or technical incident."
Output: 6

Input: "The processor implements measures pursuant to Article 32 to ensure security of processing."
Output: 6

Input: "In the event any such request is made directly to processor, processor shall notify controller in writing of such request promptly upon receipt thereof."
Output: 7

Input: "The processor may add to, amend, or replace the specific security measures for security reasons and shall notify the controller in writing where it has done so."
Output: 8

Input: "Prior consultation with the Supervisory Authority;"
Output: 9

Input: "If the Controller has an obligation to notify or report in the event of a security incident, the Processor is obliged to support the Controller at the Controllers expense."
Output: 10

Input: "Notify all Customers of any information security breach or incident that may compromise the Personal Data without undue delay after becoming aware of any such incident."
Output: 10

Input: "The processor shall assist the controller in communicating personal data breaches to affected individuals."
Output: 10

Input: "Processor shall assist Controller with any notifications to data subjects and/or authorities if requested by Controller."
Output: 10

Input: "Processor shall provide reasonable assistance to Controller with any data protection impact assessments."
Output: 11

Input: "The processor shall work with the controller to carry out a risk assessment and allow them to oversee and assess any corrective action."
Output: 12

Input: "processor will delete controller Data when requested by controller by using the Service controls provided for this purpose by processor."
Output: 13

Input: "processor shall inform controller if, in the opinion of processor, an instruction infringes Applicable Data Protection Law."
Output: 14

Input: "processor will make available to controller all information necessary to demonstrate compliance with the obligations of Data Processors laid down in Article 28 of GDPR."
Output: 15

Input: "Processor shall grant Controller access to all information required in order to verify that the obligations set out in the DPA are complied with."
Output: 16

Input: "Processor will ensure that Sub-processors are bound by written agreements that require them to provide at least the level of data protection required of Processor by these GDPR Terms."
Output: 17

Input: "processor shall remain fully liable to controller for the performance of the Sub-Processor's obligations."
Output: 18

Input: "processor has implemented and will maintain appropriate technical and organizational security measures for the Processing of Personal Data."
Output: 19

Input: "The processor implements pseudonymization and encryption of personal data to ensure security of processing."
Output: 6

Input: "The processor ensures ongoing confidentiality, integrity, availability and resilience of processing systems and services."
Output: 6

Input: "The processor maintains the ability to restore availability and access to personal data in a timely manner in the event of a physical or technical incident."
Output: 6

Input: "The processor implements measures pursuant to Article 32 to ensure security of processing."
Output: 6

**EXAMPLES (NEGATIVE/VIOLATION CASES):**

Input: "The processor shall engage sub-processors without any authorization from the controller."
Output: 1

Input: "The processor shall not inform the controller of any intended changes concerning sub-processors."
Output: 2

Input: "The processor may process personal data without documented instructions from the controller."
Output: 3

Input: "The processor shall not inform the controller of legal requirements before processing."
Output: 4

Input: "Personnel are not required to maintain confidentiality of personal data."
Output: 5

Input: "The processor shall not implement security measures for personal data processing."
Output: 6

Input: "The processor shall not assist the controller with data subject rights requests."
Output: 7

Input: "The processor shall not assist the controller in ensuring security of processing."
Output: 8

Input: "The processor shall not assist with breach notifications to supervisory authorities."
Output: 9

Input: "The processor shall not assist in communicating breaches to data subjects."
Output: 10

Input: "The processor shall not assist with data protection impact assessments."
Output: 11

Input: "The processor shall not assist in consulting supervisory authorities."
Output: 12

Input: "After the processing services have ended, the processor will not delete the data."
Output: 13

Input: "The processor shall not inform the controller of instruction infringements."
Output: 14

Input: "The processor shall not make compliance information available to the controller."
Output: 15

Input: "Information necessary for compliance demonstration will be withheld."
Output: 15

**EXAMPLES (NON-RELEVANT):**

Input: "This DPA applies when controller Data is processed by processor."
Output: NONE

Input: "The subject matter of the data processing under this DPA is controller Data."
Output: NONE

Input: "Additional instructions outside the scope of the Documented Instructions (if any) require prior written agreement between processor"
Output: NONE

Input: "processor Agreement"
Output: NONE

Input: "Appendix A"
Output: NONE

Input: "The European Parliament and the Council's Directive 95/46/EF of 24 October 1995 on the protection of individuals with regard to the processing of personal data"
Output: NONE

Input: "The processor shall comply with applicable data protection laws"
Output: NONE

Input: "The parties shall update sub-appendix A whenever changes occur that necessitates an update."
Output: NONE

Input: "Security measures are described in Appendix B."
Output: NONE

Input: "processor has established a password policy that prohibits the sharing of passwords, governs responses to password disclosure, and requires passwords to be changed on a regular basis and default passwords to be altered."
Output: NONE

Input: "The processor's liability for breach of contract shall be limited to annual fees"
Output: NONE

Input: "The Main Agreement's regulation of breach of contract and consequences shall apply"
Output: NONE

Input: "The processor maintains network firewalls and intrusion detection systems"
Output: NONE

Input: "General security policies and access controls are maintained by the processor"
Output: NONE
"""
    user_prompt = segment_text
    
    try:
        response = llm_client.generate(user_prompt, model_name=model, system_prompt=classification_system_prompt)
        # Filter out thinking sections
        response = filter_think_sections(response)
        classified_id = response.strip()
        
        if verbose:
            print(f"Classification response: {classified_id}")
        
        # Validate response
        if classified_id == "NONE":
            return "NONE"
        elif classified_id in requirements:
            return classified_id
        else:
            if verbose:
                print(f"Warning: Invalid classification response '{classified_id}', returning 'NONE'")
            return "NONE"
            
    except Exception as e:
        print(f"Error in classification: {e}")
        return "NONE"


def extract_facts_from_dpa(segment_text: str, req_text: str, req_symbolic: str, req_predicates: List[str], 
                           llm_client: OllamaClient, model: str, req_id: str = None) -> Dict:
    """Extract facts from a DPA segment using enhanced prompts."""

    # --- REFACTORED AND STRUCTURED PROMPT (FULL VERSION) ---
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
    - `ensure_security_of_processing`: ONLY extract for:
      * Explicit pseudonymization or encryption of personal data
      * Measures ensuring confidentiality/integrity/availability/resilience
      * Backup/restore capabilities for personal data availability
      * Testing specifically of Article 32 measures
    - DO NOT extract for: general IT security, logging, patches, or penetration testing
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
  REQUIREMENT: The processor shall ensure that persons authorised to process personal data are bound by confidentiality.
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
    
    try:
        response = llm_client.generate(user_prompt, model_name=model, system_prompt=fact_extraction_system_prompt)
        # Filter out thinking sections
        response = filter_think_sections(response)
        response = response.strip()
        
        if response == "NO_FACTS":
            return {}
            
        # Parse the response into a dictionary of facts
        facts = {}
        for pred in response.split(';'):
            pred = pred.strip()
            if pred and pred != "NO_FACTS" and pred != "NO_FACTS.": 
                if pred.startswith('-'):
                    facts[pred[1:]] = False
                else:
                    facts[pred] = True
        return facts
        
    except Exception as e:
        print(f"Error in fact extraction: {e}")
        return {}


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


def generate_lp_file(segment_text: str, req_text: str, req_symbolic: str, facts: Dict, req_predicates: List[str]) -> str:
    """Generate LP file content matching the existing format."""
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


def process_dpa_segments(segments_df: pd.DataFrame, requirements: Dict, llm_client: OllamaClient, 
                        model: str, output_dir: str, verbose: bool = False) -> None:
    """Process all DPA segments using RCV approach and generate LP files compatible with existing evaluation."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each segment
    for idx, row in tqdm(segments_df.iterrows(), total=len(segments_df), desc="Processing segments"):
        segment_id = row["ID"]
        segment_text = row["Sentence"]
        
        if verbose:
            print(f"\nProcessing segment {segment_id}: {segment_text[:100]}...")
        
        # Step 1: Classification
        classified_id = classify_segment(segment_text, requirements, llm_client, model, verbose)
        
        # Generate LP files for all requirements (to maintain compatibility with evaluation)
        for req_id, requirement_info in requirements.items():
            req_text = requirement_info["text"]
            req_symbolic = requirement_info["symbolic"]
            req_predicates = requirement_info["atoms"]
            
            # Create requirement directory
            req_dir = os.path.join(output_dir, f"req_{req_id}")
            os.makedirs(req_dir, exist_ok=True)
            
            if req_id == classified_id:
                # Step 2: Verification for the classified requirement
                facts = extract_facts_from_dpa(segment_text, req_text, req_symbolic, req_predicates, 
                                             llm_client, model, req_id)
                
                # Generate LP file content
                lp_content = generate_lp_file(segment_text, req_text, req_symbolic, facts, req_predicates)
            else:
                # For non-classified requirements, generate "not_mentioned" LP file
                lp_content = f"""% Requirement Text:
% {req_text}
%
% DPA Segment:
% {segment_text}
%

% RCV Classification: This segment was not classified as relevant to this requirement
% Classified as: {classified_id if classified_id != "NONE" else "Administrative/Non-relevant"}
status(not_mentioned) :- true.

#show status/1.
"""
            
            # Save LP file
            lp_file_path = os.path.join(req_dir, f"segment_{segment_id}.lp")
            with open(lp_file_path, 'w') as f:
                f.write(lp_content)
        
        if verbose:
            print(f"Generated LP files for segment {segment_id}")
            print(f"Classification: {classified_id}")
            if classified_id != "NONE":
                print(f"Verified requirement {classified_id}")


def main():
    """Main function."""
    args = parse_arguments()
    
    print("========== RCV LP File Generator ==========")
    print(f"Target DPA: {args.target_dpa}")
    print(f"Model: {args.model}")
    print(f"Output Directory: {args.output}")
    print("==========================================")
    
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
    print("Processing segments with RCV approach...")
    process_dpa_segments(
        segments_df, requirements, llm_client, args.model, 
        args.output, args.verbose
    )
    
    print(f"\nLP file generation completed!")
    print(f"Generated {len(segments_df)} LP files in: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 
