import re
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JSONLReassembler:
    def __init__(
        self,
        input_file: Path,
        output_dir: Path,
        add_page_markers: bool = True,
        skip_headers_footers: bool = False,
        output_jsonl: Optional[Path] = None,
    ):
        self.input_file = input_file
        self.output_dir = output_dir
        self.add_page_markers = add_page_markers
        self.skip_headers_footers = skip_headers_footers
        self.output_jsonl = output_jsonl
        self.stats = defaultdict(int)
        self.failed_pages = []
        self.last_failure_reason: Optional[str] = None

    def parse_page_line(self, line: str) -> Optional[Dict[str, Any]]:
        self.last_failure_reason = None
        try:
            data = json.loads(line.strip())

            if not data.get('success', False):
                logger.warning(f"Page {data.get('document_id', 'unknown')}/{data.get('page', '?')} marked as unsuccessful")
                return None

            content_str = data.get('content')
            if content_str:
                page_id = f"{data.get('document_id', 'unknown')}/{data.get('page', '?')}"
                data['content_parsed'] = self._parse_content_json(
                    content_str, page_id)
                if data['content_parsed'] is None:
                    return None

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSONL line: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing line: {e}")
            return None

    def _parse_content_json(self, content_str: str, page_id: str) -> Optional[List[Dict[str, Any]]]:
        stripped = content_str.strip()

        if self._looks_truncated(stripped):
            self.last_failure_reason = "content appears truncated"
            logger.error(f"Content JSON for {page_id} appears truncated; skipping page.")
            return None

        last_error: Optional[json.JSONDecodeError] = None

        try:
            return json.loads(content_str)
        except json.JSONDecodeError as e:
            last_error = e

        try:
            return json.loads(content_str, strict=False)
        except json.JSONDecodeError as e:
            last_error = e

        sanitized_content, modified = self._sanitize_content_string(
            content_str)
        base_content = sanitized_content if modified else content_str

        if modified:
            try:
                return json.loads(base_content)
            except json.JSONDecodeError as e:
                last_error = e
            try:
                return json.loads(base_content, strict=False)
            except json.JSONDecodeError as e:
                last_error = e

        fixed_content = base_content

        def fix_escapes(match):
            escaped_char = match.group(1)
            if escaped_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
                return match.group(0)
            return '\\\\' + escaped_char

        try:
            fixed_content = re.sub(r'\\(.)', fix_escapes, base_content)
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            pass

        try:
            fixed_content = ''.join(char if ord(
                char) >= 32 or char in '\n\r\t' else ' ' for char in base_content)
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            pass

        try:
            fixed_content = base_content
            fixed_content = re.sub(r'\\(.)', fix_escapes, fixed_content)
            fixed_content = ''.join(char if ord(
                char) >= 32 or char in '\n\r\t' else ' ' for char in fixed_content)
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            pass

        if last_error:
            self.last_failure_reason = f"JSON parsing error ({last_error.msg})"
            logger.error(f"Failed to parse content JSON for {page_id}: {last_error}")
        else:
            self.last_failure_reason = "JSON parsing failed after recovery attempts"
            logger.error(f"Failed to parse content JSON for {page_id} after all recovery attempts")
        return None

    def _sanitize_content_string(self, content_str: str) -> Tuple[str, bool]:
        chars: List[str] = []
        modified = False
        in_string = False
        escaped = False
        length = len(content_str)
        i = 0

        while i < length:
            ch = content_str[i]

            if escaped:
                chars.append(ch)
                escaped = False
                i += 1
                continue

            if ch == '\\':
                chars.append(ch)
                escaped = True
                i += 1
                continue

            if in_string:
                if ch in ('\n', '\r'):
                    chars.extend(['\\', 'n' if ch == '\n' else 'r'])
                    modified = True
                    i += 1
                    continue

                if ch == '"':
                    j = i + 1
                    while j < length and content_str[j].isspace():
                        j += 1
                    if j < length and content_str[j] not in ',:}]':
                        chars.extend(['\\', '"'])
                        modified = True
                        i += 1
                        continue
                    chars.append('"')
                    in_string = False
                    i += 1
                    continue

                chars.append(ch)
                i += 1
                continue

            else:
                if ch == '"':
                    in_string = True
                chars.append(ch)
                i += 1
                continue

        return ''.join(chars), modified

    def _looks_truncated(self, content_str: str) -> bool:
        if not content_str:
            return True

        if content_str[0] != '[' or not content_str.endswith(']'):
            return True

        bracket_depth = 0
        brace_depth = 0
        in_string = False
        escaped = False

        for ch in content_str:
            if escaped:
                escaped = False
                continue

            if ch == '\\':
                escaped = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == '[':
                bracket_depth += 1
            elif ch == ']':
                bracket_depth -= 1
                if bracket_depth < 0:
                    return True
            elif ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth < 0:
                    return True

        return bracket_depth != 0 or brace_depth != 0

    def format_formula(self, text: str) -> str:
        text = text.strip()

        if text.startswith('$$') and text.endswith('$$'):
            return text
        if text.startswith('$') and text.endswith('$'):
            return text

        if '\n' in text or text.count('\\') > 3:
            return f"$$\n{text}\n$$"
        else:
            return f"${text}$"

    def format_table(self, html_text: str) -> str:
        return f"\n{html_text}\n"

    def format_element(self, element: Dict[str, Any], page_num: int) -> str:
        category = element.get('category', 'Text')
        text = element.get('text', '').strip()

        if not text:
            return ''

        if self.skip_headers_footers and category in ['Page-header', 'Page-footer']:
            return ''

        if category == 'Title':
            return f"# {text}\n\n"

        elif category == 'Section-header':
            return f"## {text}\n\n"

        elif category == 'Formula':
            return f"{self.format_formula(text)}\n\n"

        elif category == 'Table':
            return self.format_table(text)

        elif category == 'List-item':
            if not text.startswith(('- ', '* ', '+ ', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                return f"- {text}\n"
            return f"{text}\n"

        elif category == 'Caption':
            return f"*{text}*\n\n"

        elif category == 'Footnote':
            return f"^[{text}]\n\n"

        elif category == 'Picture':
            return ''

        elif category in ['Page-header', 'Page-footer']:
            return f"_{text}_\n\n"

        else:
            return f"{text}\n\n"

    def assemble_page(self, page_data: Dict[str, Any]) -> str:
        content_elements = page_data.get('content_parsed', [])
        page_num = page_data.get('page', 0)

        markdown_parts = []

        if self.add_page_markers:
            markdown_parts.append(f"<!-- Page {page_num} -->\n\n")

        for element in content_elements:
            formatted = self.format_element(element, page_num)
            if formatted:
                markdown_parts.append(formatted)

        return ''.join(markdown_parts)

    def group_pages_by_document(self) -> Dict[str, List[Dict[str, Any]]]:
        logger.info(f"Reading and grouping pages from {self.input_file}")

        documents = defaultdict(list)
        line_count = 0
        successful_pages = 0
        failed_pages = 0

        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line_count += 1
                if not line.strip():
                    continue

                doc_id = 'unknown'
                page_num = '?'
                try:
                    quick_data = json.loads(line.strip())
                    doc_id = quick_data.get('document_id', 'unknown')
                    page_num = quick_data.get('page', '?')
                except:
                    pass

                page_data = self.parse_page_line(line)

                if page_data:
                    doc_id = page_data.get('document_id', 'unknown')
                    documents[doc_id].append(page_data)
                    successful_pages += 1
                else:
                    failed_pages += 1
                    self.failed_pages.append({
                        'document_id': doc_id,
                        'page': page_num,
                        'line': line_num,
                        'reason': self.last_failure_reason
                    })

        self.stats['total_lines'] = line_count
        self.stats['successful_pages'] = successful_pages
        self.stats['failed_pages'] = failed_pages

        logger.info(f"Grouped {successful_pages} pages into {len(documents)} documents")
        logger.info(f"Failed to parse {failed_pages} pages")

        return documents

    def assemble_document(self, doc_id: str, pages: List[Dict[str, Any]], include_header: bool = True) -> Optional[str]:
        pages_sorted = sorted(pages, key=lambda p: p.get('page', 0))
        markdown_parts = []

        if include_header:
            markdown_parts.append(f"# Document: {doc_id}\n\n")
            markdown_parts.append(f"*Processed from {len(pages_sorted)} pages*\n\n")
            markdown_parts.append("---\n\n")

        for page_data in pages_sorted:
            page_markdown = self.assemble_page(page_data)
            markdown_parts.append(page_markdown)

        return ''.join(markdown_parts)

    def process_all_documents(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        documents = self.group_pages_by_document()

        if not documents:
            logger.error("No documents found in JSONL file")
            return

        logger.info(f"Processing {len(documents)} documents")

        successful_docs = 0
        failed_docs = 0

        jsonl_file = None
        if self.output_jsonl:
            jsonl_file = open(self.output_jsonl, 'w', encoding='utf-8')
            logger.info(f"Writing JSONL output to {self.output_jsonl}")

        try:
            for doc_id in tqdm(sorted(documents.keys()), desc="Processing documents"):
                try:
                    pages = documents[doc_id]
                    markdown_content = self.assemble_document(
                        doc_id, pages, include_header=True)

                    if markdown_content:
                        output_file = self.output_dir / f"{doc_id}.md"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(markdown_content)

                        if jsonl_file:
                            markdown_no_header = self.assemble_document(
                                doc_id, pages, include_header=False)
                            jsonl_entry = {
                                "document_id": doc_id,
                                "markdown_text": markdown_no_header
                            }
                            jsonl_file.write(json.dumps(
                                jsonl_entry, ensure_ascii=False) + '\n')

                        successful_docs += 1
                    else:
                        failed_docs += 1
                        logger.warning(f"Failed to assemble document {doc_id}")

                except Exception as e:
                    failed_docs += 1
                    logger.error(f"Error processing document {doc_id}: {e}")

        finally:
            if jsonl_file:
                jsonl_file.close()

        self.stats['successful_docs'] = successful_docs
        self.stats['failed_docs'] = failed_docs

        self.generate_report()

    def generate_report(self) -> None:
        total_attempted = self.stats['successful_pages'] + \
            self.stats['failed_pages']
        success_rate = (
            100 * self.stats['successful_pages'] / total_attempted) if total_attempted > 0 else 0

        report = [
            "\n" + "="*70,
            "Processing Summary",
            "="*70,
            f"Input file: {self.input_file.name}",
            f"Total lines in JSONL: {self.stats['total_lines']}",
            f"Successful pages: {self.stats['successful_pages']}",
            f"Failed pages: {self.stats['failed_pages']}",
            f"Success rate: {success_rate:.2f}%",
            f"Documents processed: {self.stats['successful_docs']}",
            f"Documents failed: {self.stats['failed_docs']}",
            "="*70,
        ]

        if self.failed_pages:
            report.append("\nFailed Pages Detail:")
            report.append("-" * 70)

            by_doc = defaultdict(list)
            for failed in self.failed_pages:
                by_doc[failed['document_id']].append(failed)

            for doc_id in sorted(by_doc.keys()):
                pages = by_doc[doc_id]
                page_info = []
                for p in sorted(pages, key=lambda x: str(x['page'])):
                    page_info.append(f"{p['page']} (line {p['line']})")
                report.append(f"  {doc_id}: pages {', '.join(page_info)}")

            report.append("-" * 70)
            report.append(f"Total failed pages: {len(self.failed_pages)}")
            report.append("")

            report.append("\nFailed JSONL Lines (for debugging):")
            report.append("-" * 70)
            report.append("Use these line numbers to inspect the JSONL file:")
            for failed in sorted(self.failed_pages, key=lambda x: x['line']):
                report.append(f"  Line {failed['line']:5d}: {failed['document_id']}/page_{failed['page']}")
            report.append("-" * 70)
            report.append(f"\nTo inspect a specific line, use:")
            report.append(f"  sed -n '<line>p' {self.input_file.name}")
        else:
            report.append("\n✓ All pages processed successfully!")

        report.append("=" * 70 + "\n")

        report_text = '\n'.join(report)
        logger.info(report_text)

        report_file = self.output_dir / "processing_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"Report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Reassemble OCR results from JSONL into markdown documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python reassemble_from_jsonl.py dots_predictions.jsonl markdown_output

  # Without page markers
  python reassemble_from_jsonl.py dots_predictions.jsonl output --no-page-markers

  # Skip headers and footers
  python reassemble_from_jsonl.py dots_predictions.jsonl output --skip-headers-footers

  # Output JSONL
  python reassemble_from_jsonl.py dots_predictions.jsonl output --output-jsonl assembled.jsonl

Note: Enhanced error handling automatically recovers from most JSON parsing issues
        """
    )

    parser.add_argument(
        '-i', '--input_file',
        type=Path,
        help='JSONL file containing page results (one page per line)'
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=Path,
        help='Directory to save reassembled markdown files'
    )

    parser.add_argument(
        '--no-page-markers',
        action='store_true',
        help='Do not add page markers in the output'
    )

    parser.add_argument(
        '--skip-headers-footers',
        action='store_true',
        help='Skip page headers and footers'
    )

    parser.add_argument(
        '--output-jsonl',
        type=Path,
        help='Output JSONL file with assembled documents (for convert_to_merged_output.py)'
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        logger.error(f"Input file does not exist: {args.input_file}")
        return 1

    reassembler = JSONLReassembler(
        input_file=args.input_file,
        output_dir=args.output_dir,
        add_page_markers=not args.no_page_markers,
        skip_headers_footers=args.skip_headers_footers,
        output_jsonl=args.output_jsonl,
    )

    reassembler.process_all_documents()

    return 0


if __name__ == '__main__':
    exit(main())
