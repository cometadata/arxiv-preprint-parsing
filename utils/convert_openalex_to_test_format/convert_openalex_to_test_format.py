#!/usr/bin/env python3
"""
Convert OpenAlex author affiliation data to test.json format.

Fetches OpenAlex records for DOIs and extracts raw affiliation strings,
converting them to match the format used in test.json.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: pip install requests")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openalex_conversion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OpenAlexConverter:
    """Converts OpenAlex API responses to test.json format."""

    BASE_URL = "https://api.openalex.org/works/"
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # Seconds to wait before retry
    # OpenAlex rate limits: 10 req/sec, 100k/day
    # Default to 8 req/sec to be conservative
    DEFAULT_REQUESTS_PER_SECOND = 8

    def __init__(
        self,
        input_file: str = "test.json",
        output_file: str = "openalex_converted.json",
        email: Optional[str] = None,
        requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.email = email
        self.rate_limit_delay = 1.0 / requests_per_second

        self.session = requests.Session()

        # Set up User-Agent for polite pool access
        user_agent = 'OpenAlexConverter/1.0'
        if email:
            user_agent += f' (mailto:{email})'
        self.session.headers.update({'User-Agent': user_agent})

        self.stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'no_affiliations': 0
        }

        # Log rate limit and polite pool status
        if email:
            logger.info(f"Using polite pool with email: {email}")
        else:
            logger.warning("No email provided - using common pool (slower response times)")
            logger.warning("Add -e/--email argument to use the polite pool")
        logger.info(f"Rate limit: {requests_per_second} requests/second")

    def fetch_openalex_record(self, doi: str) -> Optional[Dict]:
        """
        Fetch a work record from OpenAlex API.

        Args:
            doi: The DOI of the work

        Returns:
            The work record as a dictionary, or None if fetch failed
        """
        url = f"{self.BASE_URL}{doi}"

        # Add mailto parameter for polite pool (alternative to User-Agent)
        params = {}
        if self.email:
            params['mailto'] = self.email

        for attempt in range(self.MAX_RETRIES):
            try:
                logger.debug(f"Fetching {doi} (attempt {attempt + 1}/{self.MAX_RETRIES})")
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    logger.warning(f"Record not found in OpenAlex: {doi}")
                    return None
                elif response.status_code == 429:
                    # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded (429) for {doi}")
                    if attempt < self.MAX_RETRIES - 1:
                        # Wait longer before retry
                        wait_time = self.RETRY_DELAY * (attempt + 1)
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    return None
                else:
                    logger.warning(f"HTTP {response.status_code} for {doi}")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_DELAY)
                        continue
                    return None

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {doi}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                    continue
                return None

        return None

    def convert_record(self, original_record: Dict, openalex_data: Dict) -> Dict:
        """
        Convert OpenAlex data to test.json format.

        Args:
            original_record: The original record from test.json (for arxiv_id and filename)
            openalex_data: The OpenAlex API response

        Returns:
            Converted record matching test.json format
        """
        # Extract title and language
        title_text = openalex_data.get('title', '') or openalex_data.get('display_name', '')
        title_lang = openalex_data.get('language', 'en')

        # Convert authorships to authors with raw affiliations
        authors = []
        for authorship in openalex_data.get('authorships', []):
            author_name = authorship.get('raw_author_name', '')
            if not author_name:
                # Fallback to display_name if raw_author_name is missing
                author_name = authorship.get('author', {}).get('display_name', '')

            # Get raw affiliation strings
            raw_affiliations = authorship.get('raw_affiliation_strings', [])

            if author_name:  # Only add author if we have a name
                authors.append({
                    'name': author_name,
                    'affiliations': raw_affiliations
                })

        # Track records with no affiliation data
        if all(not author['affiliations'] for author in authors):
            self.stats['no_affiliations'] += 1
            logger.info(f"No affiliation data for any author in {original_record.get('arxiv_id')}")

        # Build the converted record
        converted = {
            'arxiv_id': original_record.get('arxiv_id', ''),
            'doi': openalex_data.get('doi', original_record.get('doi', '')),
            'title': {
                'text': title_text,
                'lang': title_lang
            },
            'authors': authors,
            'filename': original_record.get('filename', '')
        }

        return converted

    def convert_all(self):
        """
        Convert all records from input file, writing results incrementally.

        Writes results to output file as each record is converted,
        rather than storing everything in memory.
        """
        # Load input file
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}")
            sys.exit(1)

        logger.info(f"Loading input from {self.input_file}")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            input_records = json.load(f)

        self.stats['total'] = len(input_records)
        logger.info(f"Processing {self.stats['total']} records...")
        logger.info(f"Writing results incrementally to {self.output_file}")

        # Open output file and write opening bracket
        with open(self.output_file, 'w', encoding='utf-8') as out_file:
            out_file.write('[\n')

            first_record = True

            for i, record in enumerate(input_records, 1):
                doi = record.get('doi', '')
                arxiv_id = record.get('arxiv_id', '')

                logger.info(f"[{i}/{self.stats['total']}] Processing {arxiv_id}")

                if not doi:
                    logger.warning(f"Skipping {arxiv_id}: no DOI found")
                    self.stats['failed'] += 1
                    continue

                # Fetch OpenAlex data
                openalex_data = self.fetch_openalex_record(doi)

                if openalex_data:
                    # Convert to target format
                    converted = self.convert_record(record, openalex_data)

                    # Write record to file immediately
                    if not first_record:
                        out_file.write(',\n')
                    else:
                        first_record = False

                    json.dump(converted, out_file, indent=2, ensure_ascii=False)
                    out_file.flush()  # Ensure it's written to disk

                    self.stats['successful'] += 1
                    logger.info(f"✓ Successfully converted {arxiv_id} ({len(converted['authors'])} authors)")
                else:
                    logger.error(f"✗ Failed to fetch OpenAlex data for {arxiv_id}")
                    self.stats['failed'] += 1

                # Rate limiting (be polite to the API)
                if i < self.stats['total']:  # Don't wait after the last request
                    time.sleep(self.rate_limit_delay)

            # Close the JSON array
            out_file.write('\n]\n')

        logger.info(f"✓ Output saved to {self.output_file}")

    def print_summary(self):
        """Print conversion statistics."""
        logger.info("\n" + "="*60)
        logger.info("CONVERSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total records:              {self.stats['total']}")
        logger.info(f"Successfully converted:     {self.stats['successful']}")
        logger.info(f"Failed:                     {self.stats['failed']}")
        logger.info(f"Records with no affiliations: {self.stats['no_affiliations']}")
        logger.info("="*60)
        if self.stats['failed'] > 0:
            logger.info("Check openalex_conversion.log for details on failures")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert OpenAlex author affiliation data to test.json format',
        epilog='OpenAlex rate limits: 10 requests/second, 100,000 requests/day. '
               'Use -e/--email to access the polite pool for better performance.'
    )
    parser.add_argument(
        '-i', '--input',
        default='test.json',
        help='Input JSON file (default: test.json)'
    )
    parser.add_argument(
        '-o', '--output',
        default='openalex_converted.json',
        help='Output JSON file (default: openalex_converted.json)'
    )
    parser.add_argument(
        '-e', '--email',
        help='Email address for OpenAlex polite pool (recommended for better performance)'
    )
    parser.add_argument(
        '-r', '--rate-limit',
        type=float,
        default=OpenAlexConverter.DEFAULT_REQUESTS_PER_SECOND,
        metavar='REQUESTS_PER_SEC',
        help=f'Rate limit in requests per second (default: {OpenAlexConverter.DEFAULT_REQUESTS_PER_SECOND}, max: 10)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate rate limit
    if args.rate_limit > 10:
        logger.warning("Rate limit exceeds OpenAlex's 10 req/sec limit. Use at your own risk!")
    if args.rate_limit <= 0:
        logger.error("Rate limit must be positive")
        sys.exit(1)

    # Run conversion
    converter = OpenAlexConverter(
        args.input,
        args.output,
        email=args.email,
        requests_per_second=args.rate_limit
    )

    try:
        converter.convert_all()
        converter.print_summary()

        # Exit with error code if any conversions failed
        if converter.stats['failed'] > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\nConversion interrupted by user")
        logger.info("Partial results have been saved to output file")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        logger.warning("Partial results may have been saved to output file")
        sys.exit(1)


if __name__ == '__main__':
    main()
