nextflow.enable.dsl=2

// --- Configuration Parameters ---
// Adjust these default paths or override them via command line (--augmented_msg_df /path/to/file)
params.scripts_dir     = "./nextflow_pipeline/nextflow_scripts"
params.outdir          = "./results"

// Input: The existing MassSpecGym dataset
params.augmented_msg_df = "/data/nas-gpu/wang/tmach007/massformer/augmented_msg_df.feather" 

// Input: The two newly created feather files (can also come from an upstream process)
// Passing them as a list for easier handling
params.new_files       = ["./path/to/new_file_1.feather", "./path/to/new_file_2.feather"]


// --- PROCESS: Remove Duplicate Spectra ---
process REMOVE_DUPLICATES {
    tag "Deduplicating new spectra"
    publishDir "${params.outdir}/deduplicated", mode: 'copy'

    input:
    path new_feather_files   // Takes the list of new files
    path existing_dataset    // Takes augmented_msg_df.feather

    output:
    path "cleaned_*.feather", emit: cleaned_feathers

    script:
    """
    python3 ${params.scripts_dir}/remove_duplicate_spectras.py \
        --new_files ${new_feather_files.join(' ')} \
        --existing_dataset ${existing_dataset} \
        --output_dir ./
    """
}

// --- PROCESS: Merge Datasets ---
process MERGE_DATASETS {
    tag "Merging datasets"
    publishDir "${params.outdir}/merged", mode: 'copy'

    input:
    path cleaned_files      // The output from REMOVE_DUPLICATES
    path original_dataset   // The original augmented_msg_df.feather

    output:
    path "merged_dataset.feather", emit: merged_file

    script:
    """
    python3 ${params.scripts_dir}/merge_datasets.py \
        --input_files ${cleaned_files.join(' ')} ${original_dataset} \
        --output_file merged_dataset.feather
    """
}

// --- WORKFLOW DEFINITION ---
workflow {
    
    // 1. Define Channels
    // Load the existing dataset
    msg_dataset_ch = Channel.fromPath(params.augmented_msg_df)
    
    // Load the new files. 
    // .collect() is used to ensure both files are passed to the process as a single list, 
    // rather than launching the process twice (once for each file).
    new_files_ch = Channel.fromPath(params.new_files).collect()

    // 2. Run Deduplication
    REMOVE_DUPLICATES(new_files_ch, msg_dataset_ch)
    
    // 3. Merge Datasets
    MERGE_DATASETS(REMOVE_DUPLICATES.out.cleaned_feathers, msg_dataset_ch)
    
    // BRUTE_FORCE_PAIRING(...)
    // SPLIT_DATA(...)
    // TRAIN_CLASSIFIER(...)
}