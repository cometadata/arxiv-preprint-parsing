import csv
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Filters a CSV file based on DOI content into three separate output files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to the input CSV file.\nExample: data.csv"
    )
    parser.add_argument(
        "-a", "--arxiv_output",
        help="Path to the output CSV file for entries with DOIs containing 'arxiv'.\n"
             "Default: <input_filename_base>_arxiv.csv"
    )
    parser.add_argument(
        "-n", "--non_arxiv_output",
        help="Path to the output CSV file for entries with DOIs not containing 'arxiv'.\n"
             "Default: <input_filename_base>_non_arxiv.csv"
    )
    parser.add_argument(
        "-x", "--no_doi_output",
        help="Path to the output CSV file for entries without DOIs.\n"
             "Default: <input_filename_base>_no_doi.csv"
    )

    args = parser.parse_args()

    input_base, _ = os.path.splitext(args.input)

    if args.arxiv_output is None:
        args.arxiv_output = f"{input_base}_arxiv.csv"
    if args.non_arxiv_output is None:
        args.non_arxiv_output = f"{input_base}_non_arxiv.csv"
    if args.no_doi_output is None:
        args.no_doi_output = f"{input_base}_no_doi.csv"

    return args


def process_csv(input_filepath, arxiv_filepath, non_arxiv_filepath, no_doi_filepath):
    processed_rows = 0
    arxiv_count = 0
    non_arxiv_count = 0
    no_doi_count = 0
    doi_key_to_use = None

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f_in, \
                open(arxiv_filepath, 'w', encoding='utf-8') as arxiv_file, \
                open(non_arxiv_filepath, 'w', encoding='utf-8') as non_arxiv_file, \
                open(no_doi_filepath, 'w', encoding='utf-8') as no_doi_file:

            reader = csv.DictReader(f_in)

            if not reader.fieldnames:
                print(f"Warning: Input file '{input_filepath}' is empty or has no header.")
                return

            for key in reader.fieldnames:
                if key.lower() == 'doi':
                    doi_key_to_use = key
                    break

            if not doi_key_to_use:
                print(f"Error: A 'doi' column (case-insensitive) not found in header: {reader.fieldnames}. Cannot process.")
                return

            writer_arxiv = csv.DictWriter(
                arxiv_file, fieldnames=reader.fieldnames)
            writer_non_arxiv = csv.DictWriter(
                non_arxiv_file, fieldnames=reader.fieldnames)
            writer_no_doi = csv.DictWriter(
                no_doi_file, fieldnames=reader.fieldnames)

            writer_arxiv.writeheader()
            writer_non_arxiv.writeheader()
            writer_no_doi.writeheader()

            for row in reader:
                processed_rows += 1
                doi_value = row.get(
                    doi_key_to_use, "").strip() if doi_key_to_use else ""

                if doi_value:
                    if "arxiv" in doi_value.lower():
                        writer_arxiv.writerow(row)
                        arxiv_count += 1
                    else:
                        writer_non_arxiv.writerow(row)
                        non_arxiv_count += 1
                else:
                    writer_no_doi.writerow(row)
                    no_doi_count += 1

            print(f"\nProcessing of '{input_filepath}' complete.")
            print(f"Total data rows processed: {processed_rows}")
            print(f" - Entries with arXiv DOIs: {arxiv_count} (saved to '{arxiv_filepath}')")
            print(f" - Entries with other DOIs: {non_arxiv_count} (saved to '{non_arxiv_filepath}')")
            print(f" - Entries without DOIs: {no_doi_count} (saved to '{no_doi_filepath}')")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    args = parse_arguments()
    process_csv(args.input, args.arxiv_output,
                args.non_arxiv_output, args.no_doi_output)


if __name__ == "__main__":
    main()
