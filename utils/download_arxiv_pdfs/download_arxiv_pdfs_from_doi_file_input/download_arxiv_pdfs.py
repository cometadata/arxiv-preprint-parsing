import os
import csv
import arxiv
import arxiv.arxiv
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Downloads arXiv PDFs from DOIs listed in CSV files."
                    "Output PDFs are saved into a directory named after the input CSV file "
                    "(e.g., 'input_file_pdfs')."
    )
    parser.add_argument(
        "-i", "--input_path", required=True,
        help="Path to a single CSV file or a directory containing CSV files. "
             "CSVs are expected to be tab-delimited with a 'doi' column."
    )
    return parser.parse_args()


def get_arxiv_id_from_doi(doi_string):
    if not isinstance(doi_string, str) or not doi_string.strip():
        return None, None

    lower_doi = doi_string.lower()

    if 'arxiv.' not in lower_doi:
        return None, None

    try:
        arxiv_id_part = lower_doi.split('arxiv.')[-1]

        if not arxiv_id_part:
            print(f"Warning: Could not extract a valid arXiv ID from '{doi_string}' after 'arxiv.'.")
            return None, None

        if arxiv_id_part.endswith('.pdf'):
            arxiv_id_part = arxiv_id_part[:-4]

        if not arxiv_id_part:
            print(f"Warning: arXiv ID became empty after sanitization for DOI '{doi_string}'.")
            return None, None

        arxiv_id_for_filename = arxiv_id_part.replace('/', '_')
        return arxiv_id_part, arxiv_id_for_filename
    except Exception as e:
        print(f"Error parsing arXiv ID from DOI '{doi_string}': {e}")
        return None, None


def download_paper_using_arxiv_library(arxiv_client, arxiv_id, arxiv_id_for_filename, output_directory):
    if not arxiv_id or not arxiv_id_for_filename:
        print("Warning: arXiv ID or filename ID is missing, skipping download.")
        return

    target_pdf_filename = f"{arxiv_id_for_filename}.pdf"
    full_output_filepath = os.path.join(output_directory, target_pdf_filename)

    if os.path.exists(full_output_filepath):
        if os.path.getsize(full_output_filepath) > 4096:
            print(f"Info: PDF already exists and is reasonably sized at '{full_output_filepath}'. Skipping download.")
            return
        else:
            print(f"Info: PDF '{full_output_filepath}' exists but is very small. Will attempt to re-download.")

    print(f"Attempting to download paper ID '{arxiv_id}' to '{full_output_filepath}' using arxiv library.")

    try:
        search = arxiv.Search(id_list=[arxiv_id])
        results_generator = arxiv_client.results(search)

        paper = next(results_generator, None)

        if paper:
            paper.download_pdf(dirpath=output_directory,
                               filename=target_pdf_filename)
            print(f"Successfully downloaded: '{full_output_filepath}'")
        else:
            print(f"Error: Paper with ID '{arxiv_id}' not found on arXiv via library search.")

    except StopIteration:
        print(f"Error: Paper with ID '{arxiv_id}' not found on arXiv (StopIteration).")
    except arxiv.arxiv.ArxivError as e:
        print(f"arXiv library error for ID '{arxiv_id}': {e}")
    except arxiv.arxiv.UnexpectedEmptyPageError as e:
        print(f"arXiv library error (UnexpectedEmptyPageError) for ID '{arxiv_id}': {e}")
    except arxiv.arxiv.HTTPError as e:
        print(f"arXiv library error (HTTPError) for ID '{arxiv_id}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred for ID '{arxiv_id}': {e}")


def process_csv_file(csv_filepath, output_directory_for_csv, arxiv_client):
    print(f"\nProcessing CSV file: '{csv_filepath}'")

    try:
        os.makedirs(output_directory_for_csv, exist_ok=True)
        print(f"Outputting PDFs to: '{output_directory_for_csv}'")
    except OSError as e:
        print(f"Error creating output directory '{output_directory_for_csv}': {e}. Skipping this CSV.")
        return

    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)

            header = next(reader, None)
            if not header:
                print(f"Warning: CSV file '{csv_filepath}' is empty or has no header. Skipping.")
                return

            try:
                normalized_header = [h.strip().lower() for h in header]
                doi_column_index = normalized_header.index('doi')
            except ValueError:
                print(f"Error: 'doi' column not found in the header of '{csv_filepath}'. Header found: {header}. Skipping this file.")
                return

            for i, row in enumerate(reader):
                if not row:
                    continue

                if len(row) <= doi_column_index:
                    print(f"Warning: Row {i+2} in '{csv_filepath}' does not have enough columns for DOI (expected index {doi_column_index}). Row: {row}. Skipping.")
                    continue

                doi_string = row[doi_column_index].strip()
                if not doi_string:
                    continue

                actual_arxiv_id, arxiv_id_for_filename = get_arxiv_id_from_doi(
                    doi_string)

                if actual_arxiv_id and arxiv_id_for_filename:
                    download_paper_using_arxiv_library(
                        arxiv_client, actual_arxiv_id, arxiv_id_for_filename, output_directory_for_csv)
                elif doi_string:
                    print(f"Info: Could not process DOI '{doi_string}' from row {i+2} in '{csv_filepath}' to get arXiv ID.")

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_filepath}'.")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{csv_filepath}': {e}")


def main():
    args = parse_arguments()
    input_path = args.input_path

    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        return

    arxiv_client = arxiv.Client()

    csv_files_to_process = []
    output_location_map = {}

    if os.path.isfile(input_path):
        if input_path.lower().endswith(".csv"):
            csv_files_to_process.append(input_path)
            file_directory = os.path.dirname(input_path)
            if not file_directory:
                file_directory = "."
            base_filename = os.path.splitext(os.path.basename(input_path))[0]
            output_dir_name = f"{base_filename}_pdfs"
            output_location_map[input_path] = os.path.join(
                file_directory, output_dir_name)
        else:
            print(f"Error: Specified input file '{input_path}' is not a .csv file.")
            return
    elif os.path.isdir(input_path):
        for item_name in os.listdir(input_path):
            full_item_path = os.path.join(input_path, item_name)
            if os.path.isfile(full_item_path) and item_name.lower().endswith(".csv"):
                csv_files_to_process.append(full_item_path)
                base_filename = os.path.splitext(item_name)[0]
                output_dir_name = f"{base_filename}_pdfs"
                output_location_map[full_item_path] = os.path.join(
                    input_path, output_dir_name)

        if not csv_files_to_process:
            print(f"No .csv files found in the directory '{input_path}'.")
            return
    else:
        print(f"Error: The input path '{input_path}' is neither a file nor a directory.")
        return

    if not csv_files_to_process:
        print("No CSV files were identified for processing.")
        return

    print(f"Found {len(csv_files_to_process)} CSV file(s) to process.")
    for csv_file_path in csv_files_to_process:
        target_output_dir = output_location_map[csv_file_path]
        process_csv_file(csv_file_path, target_output_dir, arxiv_client)

    print("\nPDF download process has finished.")


if __name__ == "__main__":
    main()
