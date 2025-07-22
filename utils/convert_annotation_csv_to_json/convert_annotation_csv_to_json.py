import csv
import json
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert arXiv annotation CSV to a structured JSON format.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("-o", "--output", required=True, help="Path for the output JSON file.")
    return parser.parse_args()


def normalize_value(value):
    if not isinstance(value, str):
        return None
    stripped_value = value.strip()
    if not stripped_value or stripped_value.lower() in ['na', 'n/a']:
        return None

    return stripped_value


def process_csv_to_json(input_path, output_path):
    publications = {}
    try:
        with open(input_path, mode='r', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)

            for row in reader:
                pdf_link = normalize_value(row.get('PDF File Link'))
                if not pdf_link:
                    continue

                if pdf_link not in publications:
                    base_id = pdf_link.removesuffix('.pdf')
                    arxiv_base_id = base_id.replace('_', '/')
                    arxiv_id = f"arXiv:{arxiv_base_id}"
                    doi_arxiv_id = f"arXiv.{arxiv_base_id}"
                    doi = f"https://doi.org/10.48550/{doi_arxiv_id}"

                    publications[pdf_link] = {
                        "arxiv_id": arxiv_id,
                        "doi": doi,
                        "title": {
                            "text": normalize_value(row.get('title')),
                            "lang": normalize_value(row.get('title_lang'))
                        },
                        "authors": [],
                        "_authors_map": {}
                    }

                author_name = normalize_value(row.get('authorName'))
                if author_name:
                    publication_entry = publications[pdf_link]
                    authors_map = publication_entry["_authors_map"]

                    if author_name not in authors_map:
                        new_author = {
                            "name": author_name,
                            "affiliations": []
                        }
                        publication_entry["authors"].append(new_author)
                        authors_map[author_name] = new_author

                    affiliation = normalize_value(
                        row.get('author_affiliation'))
                    if affiliation:
                        if affiliation not in authors_map[author_name]["affiliations"]:
                            authors_map[author_name]["affiliations"].append(
                                affiliation)

    except FileNotFoundError:
        print(f"Error: The input file was not found at {input_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    output_list = []
    for pdf_link in publications:
        del publications[pdf_link]["_authors_map"]
        output_list.append(publications[pdf_link])

    try:
        with open(output_path, 'w', encoding='utf-8') as json_out:
            json.dump(output_list, json_out, indent=2, ensure_ascii=False)
        print(f"Successfully converted {input_path} to {output_path}")
    except IOError:
        print(f"Error: Could not write to the output file at {output_path}")


def main():
    args = parse_arguments()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at '{args.input}'")
        return

    process_csv_to_json(args.input, args.output)


if __name__ == "__main__":
    main()
