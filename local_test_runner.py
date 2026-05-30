import os
import sys
import json
from agents.orchestrator_agent.orchestrator_agent import orchestrate_full_pipeline

def run_test():
    print("=" * 60)
    print("STARTING LOCAL MULTI-AGENT DATA ANALYTICS PIPELINE TEST")
    print("=" * 60)
    
    # We will query the pipeline to scrape example.com which is fast and simple
    test_query = "Scrape data from https://example.com and analyze it"
    print(f"User Query: {test_query}\n")
    
    try:
        # Run the full end-to-end pipeline
        result = orchestrate_full_pipeline(user_query=test_query, start_from_csv_path=None)
        
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTED SUCCESSFULLY")
        print("=" * 60)
        
        # Verify the artifacts
        artifact_paths = result.get("artifact_paths", {})
        
        print("\nVerifying Generated Artifacts:")
        expected_artifacts = [
            ("raw_data_path", "Raw Data JSON"),
            ("extracted_dataframe_path", "Extracted CSV"),
            ("cleaned_dataframe_path", "Cleaned CSV"),
            ("engineered_dataframe_path", "Feature Engineered CSV"),
            ("eda_metadata_path", "EDA Metadata JSON"),
            ("dashboard_html_path", "BI Dashboard HTML"),
            ("orchestration_metadata_path", "Orchestrator Metadata JSON")
        ]
        
        all_present = True
        for key, description in expected_artifacts:
            path = artifact_paths.get(key)
            if not path:
                print(f"[-] {description}: MISSING from result dict")
                all_present = False
            elif not os.path.exists(path):
                print(f"[-] {description}: File does not exist at {path}")
                all_present = False
            elif os.path.getsize(path) == 0:
                print(f"[-] {description}: File at {path} is empty (0 bytes)")
                all_present = False
            else:
                print(f"[+] {description}: Created successfully ({os.path.getsize(path)} bytes)")
                print(f"    Path: {path}")
        
        if all_present:
            print("\n[SUCCESS] All pipeline stages ran successfully and all expected output artifacts exist!")
        else:
            print("\n[FAILURE] One or more expected output artifacts are missing or empty.")
            sys.exit(1)
            
        print("\nPipeline Stage Logs:")
        for log in result.get("logs", []):
            print(f" - {log}")
            
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()
