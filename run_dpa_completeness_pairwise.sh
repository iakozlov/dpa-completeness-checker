#!/bin/bash
# run_dpa_completeness_pairwise.sh
# Master script to run the DPA completeness evaluation pipeline with Pairwise approach

set -e  # Exit on any error

# Configuration
DPA_CSV="data/test_set.csv"
REQUIREMENTS_FILE="data/requirements/requirements_deontic_manual.json"
OLLAMA_MODEL="qwen2.5:14b"  # Default Ollama model
OUTPUT_DIR="results/pairwise_approach/initial_reqs/qwen2.5-14b"
DEOLINGO_RESULTS="${OUTPUT_DIR}/deolingo_results.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.json"
TARGET_DPAS=("Online 124" "Online 126" "Online 54" "Online 57" "Online 39" "Online 140"  "Online 87" )  # Array of DPAs to process
REQ_IDS="all"  # Focus on all requirements by default
MAX_SEGMENTS=0  # Process all segments by default
REQUIREMENTS_REPRESENTATION="deontic_ai"  # Default representation

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            OLLAMA_MODEL="$2"
            shift 2
            ;;
        --requirements)
            REQUIREMENTS_REPRESENTATION="$2"
            shift 2
            ;;
        --req_ids)
            REQ_IDS="$2"
            shift 2
            ;;
        --max_segments)
            MAX_SEGMENTS="$2"
            shift 2
            ;;
        --target_dpas)
            IFS=',' read -ra TARGET_DPAS <<< "$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Function to log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if Ollama is running
check_ollama() {
    local ollama_url="${OLLAMA_HOST:-http://localhost:11434}"
    if ! curl -f -s "${ollama_url}/api/tags" > /dev/null; then
        log "ERROR: Ollama server is not running at ${ollama_url}!"
        log "Please start Ollama server first"
        exit 1
    fi
    log "Ollama server is running at ${ollama_url}"
}

# Function to pull model if not available
ensure_model() {
    local model=$1
    log "Checking if model ${model} is available..."
    
    # If OLLAMA_HOST is set to a remote server, assume model management is handled externally
    if [[ "${OLLAMA_HOST:-}" =~ ^http://ollama: ]]; then
        log "Using remote Ollama server, assuming model ${model} is available"
        return 0
    fi
    
    if ! ollama list | grep -q "${model}"; then
        log "Model ${model} not found locally. Pulling from registry..."
        if ! ollama pull "${model}"; then
            log "ERROR: Failed to pull model ${model}"
            exit 1
        fi
        log "Successfully pulled model ${model}"
    else
        log "Model ${model} is available"
    fi
}

# Function to set requirements file based on representation
set_requirements_file() {
    case $REQUIREMENTS_REPRESENTATION in
        "deontic")
            REQUIREMENTS_FILE="data/requirements/requirements_deontic.json"
            ;;
        "deontic_ai")
            REQUIREMENTS_FILE="data/requirements/requirements_deontic_manual.json"
            ;;
        "deontic_short")
            REQUIREMENTS_FILE="data/requirements/requirements_deontic_short.json"
            ;;
        "deontic_verbose")
            REQUIREMENTS_FILE="data/requirements/requirements_deontic_verbose.json"
            ;;
        *)
            log "ERROR: Unknown requirements representation: $REQUIREMENTS_REPRESENTATION"
            log "Available options: deontic, deontic_ai, deontic_short, deontic_verbose"
            exit 1
            ;;
    esac
    
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        log "ERROR: Requirements file not found: $REQUIREMENTS_FILE"
        exit 1
    fi
    
    log "Using requirements file: $REQUIREMENTS_FILE"
}

# Function to run deolingo with error handling
run_deolingo() {
    local lp_file=$1
    local req_id=$2
    local segment_id=$3
    local dpa_name=$4
    
    # Run deolingo and capture output
    deolingo_output=$(deolingo ${lp_file} 2>&1) || true
    
    # Check if there was an error
    if [[ $deolingo_output == *"ERROR"* || $deolingo_output == *"error"* ]]; then
        echo "Processing DPA ${dpa_name}, Requirement ${req_id}, Segment ${segment_id}..." >> ${DEOLINGO_RESULTS}
        echo "Error processing file: ${lp_file}" >> ${DEOLINGO_RESULTS}
        echo "Answer: not_mentioned" >> ${DEOLINGO_RESULTS}
        echo "--------------------------------------------------" >> ${DEOLINGO_RESULTS}
        echo "Warning: Error in file ${lp_file}. Skipping and continuing..." >&2
    else
        echo "Processing DPA ${dpa_name}, Requirement ${req_id}, Segment ${segment_id}..." >> ${DEOLINGO_RESULTS}
        echo "${deolingo_output}" >> ${DEOLINGO_RESULTS}
        echo "--------------------------------------------------" >> ${DEOLINGO_RESULTS}
    fi
}

# Function to check if deolingo is installed
check_deolingo() {
    if ! command -v deolingo &> /dev/null; then
        log "ERROR: deolingo command not found!"
        log "Deolingo is required to run the solver (steps 2-3)."
        log ""
        log "To install deolingo on your server, run:"
        log "  pip3 install git+https://github.com/ovidiomanteiga/deolingo.git@main"
        log ""
        log "Or install from the local repository:"
        log "  cd $(pwd) && pip3 install -e ./deolingo/"
        log ""
        log "For detailed installation instructions, see:"
        log "  docs/DEOLINGO_SERVER_SETUP.md"
        log ""
        log "After installation, make sure deolingo is in your PATH:"
        log "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        exit 1
    fi
    log "Deolingo is available"
}

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set requirements file
set_requirements_file

# Update output paths based on configuration
MODEL_SAFE_NAME=$(echo "${OLLAMA_MODEL}" | tr ':' '_' | tr '/' '_')
DEOLINGO_RESULTS="${OUTPUT_DIR}/deolingo_results_${MODEL_SAFE_NAME}_pairwise.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results_${MODEL_SAFE_NAME}_pairwise.json"

echo "========== DPA Completeness Checker - Pairwise Approach =========="
echo "Using Ollama Model: ${OLLAMA_MODEL}"
echo "Requirements Representation: ${REQUIREMENTS_REPRESENTATION}"
echo "Requirements Source: ${REQUIREMENTS_FILE}"
echo "Evaluating DPAs: ${TARGET_DPAS[*]}"
echo "Focus on requirement(s): ${REQ_IDS}"
echo "Using ${MAX_SEGMENTS} segments (0 means all)"
echo "Output Directory: ${OUTPUT_DIR}"
echo "========================================================"

# Show menu of available steps
echo "Available steps:"
echo "1. Generate Pairwise LP files for specified requirements and segments for all DPAs"
echo "2. Run Deolingo solver for all DPAs"
echo "3. Evaluate DPA completeness (Requirement & Segment-level metrics)"
echo "A. Run all steps sequentially"
echo "Q. Quit"

read -p "Enter step to run (1-3, A for all, Q to quit): " STEP

case ${STEP} in
    1)
        echo -e "\n[Step 1] Generating Pairwise LP files for requirement(s) ${REQ_IDS} and ${MAX_SEGMENTS} segments..."
        
        # Check if requirements file exists
        if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
            log "ERROR: Requirements file not found: $REQUIREMENTS_FILE"
            exit 1
        fi
        
        # Check Ollama server and model availability
        check_ollama
        ensure_model "$OLLAMA_MODEL"
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            echo "Processing DPA: ${TARGET_DPA}"
            python3 generate_lp_files.py \
              --requirements "${REQUIREMENTS_FILE}" \
              --dpa ${DPA_CSV} \
              --model ${OLLAMA_MODEL} \
              --output "${OUTPUT_DIR}/lp_files_${TARGET_DPA//' '/_}" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments ${MAX_SEGMENTS} \
              --req_ids "${REQ_IDS}" \
              --verbose
        done
        echo "Step 1 completed. Pairwise LP files generated in: ${OUTPUT_DIR}/lp_files_[DPA_NAME]"
        ;;
    2)
        echo -e "\n[Step 2] Running Deolingo solver for all DPAs..."
        
        # Check if deolingo is installed
        check_deolingo
        
        # Create output file for results
        echo "" > ${DEOLINGO_RESULTS}
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            DPA_DIR="${OUTPUT_DIR}/lp_files_${TARGET_DPA//' '/_}/dpa_${TARGET_DPA//' '/_}"
            
            if [ ! -d "${DPA_DIR}" ]; then
                echo "Error: LP files directory not found at ${DPA_DIR} for DPA ${TARGET_DPA}. Run Step 1 first."
                continue
            fi
            
            echo "Processing DPA: ${TARGET_DPA}"
            
            # Process requirements - use only those specified by REQ_IDS if not "all"
            if [ "${REQ_IDS}" == "all" ]; then
                REQ_DIRS=$(find "${DPA_DIR}" -type d -name "req_*")
            else
                REQ_DIRS=""
                for REQ_ID in $(echo ${REQ_IDS} | tr ',' ' '); do
                    if [ -d "${DPA_DIR}/req_${REQ_ID}" ]; then
                        REQ_DIRS="${REQ_DIRS} ${DPA_DIR}/req_${REQ_ID}"
                    fi
                done
            fi
            
            for req_dir in ${REQ_DIRS}; do
                req_id=$(basename ${req_dir} | sed 's/req_//')
                echo "  Processing requirement ${req_id}..."
                
                # Process all .lp files for this requirement
                find "${req_dir}" -name "*.lp" | while read lp_file; do
                    # Extract segment ID from the file path
                    segment_id=$(basename $lp_file | sed 's/segment_//' | sed 's/\.lp//')
                    
                    # Run deolingo with error handling
                    run_deolingo "${lp_file}" "${req_id}" "${segment_id}" "${TARGET_DPA}"
                done
            done
        done
        echo "Step 2 completed. Results saved in: ${DEOLINGO_RESULTS}"
        ;;
    3)
        echo -e "\n[Step 3] Evaluating DPA completeness (aggregated results)..."
        if [ ! -f "${DEOLINGO_RESULTS}" ]; then
            echo "Error: Deolingo results file not found. Run Step 2 first."
            exit 1
        fi
        
        # Process each DPA and then aggregate results
        TEMP_OUTPUTS=""
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            TEMP_OUTPUT="${OUTPUT_DIR}/evaluation_${TARGET_DPA//' '/_}.json"
            if [ -z "$TEMP_OUTPUTS" ]; then
                TEMP_OUTPUTS="${TEMP_OUTPUT}"
            else
                TEMP_OUTPUTS="${TEMP_OUTPUTS},${TEMP_OUTPUT}"
            fi
            
            echo "Processing DPA: ${TARGET_DPA}"
            python3 evaluate_completeness.py \
              --results ${DEOLINGO_RESULTS} \
              --dpa ${DPA_CSV} \
              --output ${TEMP_OUTPUT} \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}"
        done
        
        # Aggregate results
        python3 aggregate_evaluations.py \
          --input_files "${TEMP_OUTPUTS}" \
          --output ${EVALUATION_OUTPUT}
        
        echo "Step 3 completed. Aggregated evaluation results saved to: ${EVALUATION_OUTPUT}"
        ;;
    A|a)
        echo -e "\nRunning all steps sequentially..."
        
        # Step 1
        echo -e "\n[Step 1] Generating Pairwise LP files for requirement(s) ${REQ_IDS} and ${MAX_SEGMENTS} segments..."
        
        # Check Ollama server and model availability
        check_ollama
        ensure_model "$OLLAMA_MODEL"
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            echo "Processing DPA: ${TARGET_DPA}"
            python3 generate_lp_files.py \
              --requirements "${REQUIREMENTS_FILE}" \
              --dpa ${DPA_CSV} \
              --model ${OLLAMA_MODEL} \
              --output "${OUTPUT_DIR}/lp_files_${TARGET_DPA//' '/_}" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments ${MAX_SEGMENTS} \
              --req_ids "${REQ_IDS}" \
              --verbose
        done
        
        # Step 2
        echo -e "\n[Step 2] Running Deolingo solver for all DPAs..."
        check_deolingo
        echo "" > ${DEOLINGO_RESULTS}
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            DPA_DIR="${OUTPUT_DIR}/lp_files_${TARGET_DPA//' '/_}/dpa_${TARGET_DPA//' '/_}"
            echo "Processing DPA: ${TARGET_DPA}"
            
            if [ "${REQ_IDS}" == "all" ]; then
                REQ_DIRS=$(find "${DPA_DIR}" -type d -name "req_*")
            else
                REQ_DIRS=""
                for REQ_ID in $(echo ${REQ_IDS} | tr ',' ' '); do
                    if [ -d "${DPA_DIR}/req_${REQ_ID}" ]; then
                        REQ_DIRS="${REQ_DIRS} ${DPA_DIR}/req_${REQ_ID}"
                    fi
                done
            fi
            
            for req_dir in ${REQ_DIRS}; do
                req_id=$(basename ${req_dir} | sed 's/req_//')
                echo "  Processing requirement ${req_id}..."
                find "${req_dir}" -name "*.lp" | while read lp_file; do
                    segment_id=$(basename $lp_file | sed 's/segment_//' | sed 's/\.lp//')
                    run_deolingo "${lp_file}" "${req_id}" "${segment_id}" "${TARGET_DPA}"
                done
            done
        done
        
        # Step 3
        echo -e "\n[Step 3] Evaluating DPA completeness..."
        TEMP_OUTPUTS=""
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            TEMP_OUTPUT="${OUTPUT_DIR}/evaluation_${TARGET_DPA//' '/_}.json"
            if [ -z "$TEMP_OUTPUTS" ]; then
                TEMP_OUTPUTS="${TEMP_OUTPUT}"
            else
                TEMP_OUTPUTS="${TEMP_OUTPUTS},${TEMP_OUTPUT}"
            fi
            
            python3 evaluate_completeness.py \
              --results ${DEOLINGO_RESULTS} \
              --dpa ${DPA_CSV} \
              --output ${TEMP_OUTPUT} \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}"
        done
        
        python3 aggregate_evaluations.py \
          --input_files "${TEMP_OUTPUTS}" \
          --output ${EVALUATION_OUTPUT}
        
        echo -e "\n========================================================"
        echo "All steps completed successfully!"
        echo "Configuration used:"
        echo "  Model: ${OLLAMA_MODEL} (Ollama)"
        echo "  Requirements representation: ${REQUIREMENTS_REPRESENTATION}"
        echo "  Requirements source: Predefined"
        echo "  Target DPAs: ${TARGET_DPAS[*]}"
        echo "  Requirement IDs: ${REQ_IDS}"
        echo "  Max segments: ${MAX_SEGMENTS}"
        echo ""
        echo "Results saved to:"
        echo "  Output directory: ${OUTPUT_DIR}"
        echo "  Deolingo results: ${DEOLINGO_RESULTS}"
        echo "  Evaluation results: ${EVALUATION_OUTPUT}"
        echo "========================================================"
        ;;
    Q|q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option. Please choose a valid step (1-3, A, or Q)."
        exit 1
        ;;
esac 
