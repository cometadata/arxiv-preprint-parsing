import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from convert_inferrence_output_to_evaluation_format import (
    normalize_arxiv_id,
    process_predicted_authors,
    convert_content_to_predictions,
    Author
)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestNormalizeArxivId(unittest.TestCase):
    def test_standard_formats(self):
        """Test standard arxiv ID formats."""
        # Standard format with lowercase prefix
        self.assertEqual(normalize_arxiv_id('arxiv:1234.5678'), 'arXiv:1234.5678')
        # Standard format with mixed case prefix
        self.assertEqual(normalize_arxiv_id('arXiv:1234.5678'), 'arXiv:1234.5678')
        # Standard format without prefix
        self.assertEqual(normalize_arxiv_id('1234.5678'), 'arXiv:1234.5678')

    def test_versioned_ids(self):
        """Test arxiv IDs with version numbers."""
        self.assertEqual(normalize_arxiv_id(
            'arxiv:1234.5678v1'), 'arXiv:1234.5678v1')
        self.assertEqual(normalize_arxiv_id(
            'arXiv:1234.5678v2'), 'arXiv:1234.5678v2')
        self.assertEqual(normalize_arxiv_id('1234.5678v3'), 'arXiv:1234.5678v3')

    def test_underscore_format(self):
        """Test underscore format conversion."""
        self.assertEqual(normalize_arxiv_id('1234_5678'), 'arXiv:1234/5678')
        self.assertEqual(normalize_arxiv_id('arxiv:1234_5678'), 'arXiv:1234/5678')

    def test_old_style_ids(self):
        """Test old-style arxiv IDs with category prefixes."""
        self.assertEqual(normalize_arxiv_id(
            'astro-ph/0123456'), 'arXiv:astro-ph/0123456')
        self.assertEqual(normalize_arxiv_id(
            'cond-mat/9901234'), 'arXiv:cond-mat/9901234')
        self.assertEqual(normalize_arxiv_id(
            'arxiv:hep-th/0123456'), 'arXiv:hep-th/0123456')

    def test_edge_cases(self):
        """Test edge cases and malformed inputs."""
        # Empty string
        self.assertEqual(normalize_arxiv_id(''), 'arXiv:')
        # Random string without arxiv pattern
        self.assertEqual(normalize_arxiv_id(
            'not-an-arxiv-id'), 'arXiv:not-an-arxiv-id')
        # Mixed content with arxiv ID embedded
        self.assertEqual(
            normalize_arxiv_id(
                'See paper at 1234.5678 for details'),
            'arXiv:1234.5678'
        )


class TestProcessPredictedAuthors(unittest.TestCase):
    def test_valid_authors(self):
        """Test processing of valid author data."""
        input_authors = [
            {'name': 'John Doe', 'affiliations': ['MIT', 'Harvard']},
            {'name': 'Jane Smith', 'affiliations': ['Stanford']}
        ]
        result = process_predicted_authors(input_authors)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['name'], 'John Doe')
        self.assertEqual(result[0]['affiliations'], ['MIT', 'Harvard'])
        self.assertEqual(result[1]['name'], 'Jane Smith')
        self.assertEqual(result[1]['affiliations'], ['Stanford'])

    def test_missing_affiliations(self):
        """Test handling of authors without affiliations."""
        input_authors = [
            {'name': 'John Doe'},
            {'name': 'Jane Smith', 'affiliations': None}
        ]
        result = process_predicted_authors(input_authors)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['affiliations'], [])
        self.assertEqual(result[1]['affiliations'], [])

    def test_invalid_input_types(self):
        """Test handling of invalid input types."""
        # None input
        self.assertEqual(process_predicted_authors(None), [])
        # Not a list
        self.assertEqual(process_predicted_authors('not a list'), [])
        # Dictionary instead of list
        self.assertEqual(process_predicted_authors({'name': 'John'}), [])

    @patch('convert_inferrence_output_to_evaluation_format.logging')
    def test_invalid_author_entries_with_logging(self, mock_logging):
        """Test that invalid entries generate appropriate warnings."""
        input_authors = [
            'not a dict',  # Invalid: string instead of dict
            {'missing_name': 'value'},  # Invalid: no 'name' field
            {'name': ''},  # Invalid: empty name
            {'name': 'Valid Author', 'affiliations': ['MIT']}  # Valid
        ]
        result = process_predicted_authors(input_authors, line_num=42)

        # Only the valid author should be in result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], 'Valid Author')

        # Check that warnings were logged
        self.assertEqual(mock_logging.warning.call_count, 3)

    def test_non_list_affiliations(self):
        """Test handling of non-list affiliations."""
        input_authors = [
            {'name': 'John Doe', 'affiliations': 'MIT'},  # String instead of list
            {'name': 'Jane Smith', 'affiliations': {
                'org': 'Stanford'}}  # Dict instead of list
        ]
        result = process_predicted_authors(input_authors)
        self.assertEqual(len(result), 2)
        # String affiliations should become single-item lists
        self.assertEqual(result[0]['affiliations'], ['MIT'])
        self.assertEqual(result[1]['affiliations'], [])


class TestConvertContentToPredictions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = Path(self.temp_dir) / 'input.jsonl'
        self.output_file = Path(self.temp_dir) / 'output.json'
        self.ground_truth_file = Path(self.temp_dir) / 'ground_truth.json'

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_basic_conversion(self):
        input_data = [
            {
                'arxiv_id': 'arXiv:1234.5678',
                'predicted_authors': [
                    {'name': 'Author One', 'affiliations': ['Univ A']},
                    {'name': 'Author Two', 'affiliations': [
                        'Univ B', 'Univ C']}
                ]
            },
            {
                'arxiv_id': 'arXiv:2345.6789',
                'predicted_authors': [
                    {'name': 'Author Three', 'affiliations': []}
                ]
            }
        ]

        # Write input file
        with open(self.input_file, 'w') as f:
            json.dump(input_data, f)

        # Run conversion
        convert_content_to_predictions(
            str(self.input_file), str(self.output_file))

        # Check output
        with open(self.output_file, 'r') as f:
            output = json.load(f)

        self.assertIn('arXiv:1234.5678', output)
        self.assertIn('arXiv:2345.6789', output)
        # Output should be wrapped in {"predicted_authors": [...]}
        self.assertIn('predicted_authors', output['arXiv:1234.5678'])
        self.assertEqual(len(output['arXiv:1234.5678']['predicted_authors']), 2)
        self.assertEqual(len(output['arXiv:2345.6789']['predicted_authors']), 1)

    def test_null_predicted_authors(self):
        """Test handling of null predicted_authors."""
        input_data = [
            {'arxiv_id': 'arXiv:1234.5678', 'predicted_authors': None},
            {'arxiv_id': 'arXiv:2345.6789', 'predicted_authors': []},
            {'arxiv_id': 'arXiv:3456.7890',
                'predicted_authors': [{'name': 'Author'}]}
        ]

        with open(self.input_file, 'w') as f:
            json.dump(input_data, f)

        convert_content_to_predictions(
            str(self.input_file), str(self.output_file))

        with open(self.output_file, 'r') as f:
            output = json.load(f)

        # All entries should be present
        self.assertEqual(len(output), 3)
        # First two should have empty lists wrapped in predicted_authors
        self.assertEqual(output['arXiv:1234.5678']['predicted_authors'], [])
        self.assertEqual(output['arXiv:2345.6789']['predicted_authors'], [])
        # Third should have one author
        self.assertEqual(len(output['arXiv:3456.7890']['predicted_authors']), 1)

    def test_with_ground_truth_matching(self):
        """Test ID format matching with ground truth file."""
        # Create ground truth with specific ID format
        ground_truth = [
            {'arxiv_id': 'arXiv:1234.5678'},
            {'arxiv_id': 'arXiv:2345.6789'}
        ]

        with open(self.ground_truth_file, 'w') as f:
            json.dump(ground_truth, f)

        # Input with different ID format
        input_data = [
            {
                'arxiv_id': '1234.5678',  # No prefix
                'predicted_authors': [{'name': 'Author One', 'affiliations': []}]
            },
            {
                'arxiv_id': 'arxiv:2345.6789',  # Lowercase prefix
                'predicted_authors': [{'name': 'Author Two', 'affiliations': []}]
            }
        ]

        with open(self.input_file, 'w') as f:
            json.dump(input_data, f)

        convert_content_to_predictions(
            str(self.input_file),
            str(self.output_file),
            str(self.ground_truth_file)
        )

        with open(self.output_file, 'r') as f:
            output = json.load(f)

        # Output should use ground truth format
        self.assertIn('arXiv:1234.5678', output)
        self.assertIn('arXiv:2345.6789', output)

    @patch('convert_inferrence_output_to_evaluation_format.logging')
    def test_error_handling(self, mock_logging):
        """Test error handling for malformed JSON."""
        # Create input with valid and invalid JSON
        with open(self.input_file, 'w') as f:
            f.write('{"arxiv_id": "1234.5678", "predicted_authors": []}\n')
            f.write('not valid json\n')
            f.write('{"arxiv_id": "2345.6789", "predicted_authors": []}\n')
            f.write('{"missing_arxiv_id": true}\n')

        convert_content_to_predictions(
            str(self.input_file), str(self.output_file))

        with open(self.output_file, 'r') as f:
            output = json.load(f)

        # Should have processed 2 valid entries with arXiv: prefix
        self.assertEqual(len(output), 2)
        self.assertIn('arXiv:1234.5678', output)
        self.assertIn('arXiv:2345.6789', output)

        # Check that errors were logged
        error_calls = mock_logging.error.call_args_list
        self.assertTrue(any('parsing JSON' in str(call)
                            for call in error_calls))
        self.assertTrue(any('missing identifiable arxiv id fields' in str(call)
                            for call in error_calls))

    def test_empty_lines_handling(self):
        """Test that empty lines are properly skipped."""
        with open(self.input_file, 'w') as f:
            f.write('{"arxiv_id": "1234.5678", "predicted_authors": []}\n')
            f.write('\n')  # Empty line
            f.write('   \n')  # Whitespace only
            f.write('{"arxiv_id": "2345.6789", "predicted_authors": []}\n')

        convert_content_to_predictions(
            str(self.input_file), str(self.output_file))

        with open(self.output_file, 'r') as f:
            output = json.load(f)

        # Should have processed 2 entries, skipping empty lines
        self.assertEqual(len(output), 2)

    def test_special_characters_in_names(self):
        """Test handling of special characters in author names."""
        input_data = [
            {
                'arxiv_id': 'arXiv:1234.5678',
                'predicted_authors': [
                    {'name': 'José García-López',
                        'affiliations': ['Universidad']},
                    {'name': 'Anne-Marie O\'Brien', 'affiliations': []},
                    {'name': 'Müller, K.', 'affiliations': ['TU München']}
                ]
            }
        ]

        with open(self.input_file, 'w') as f:
            json.dump(input_data, f)

        convert_content_to_predictions(
            str(self.input_file), str(self.output_file))

        with open(self.output_file, 'r') as f:
            output = json.load(f)

        authors = output['arXiv:1234.5678']['predicted_authors']
        self.assertEqual(len(authors), 3)
        self.assertEqual(authors[0]['name'], 'José García-López')
        self.assertEqual(authors[1]['name'], 'Anne-Marie O\'Brien')
        self.assertEqual(authors[2]['name'], 'Müller, K.')


class TestStatisticsTracking(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = Path(self.temp_dir) / 'input.jsonl'
        self.output_file = Path(self.temp_dir) / 'output.json'

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('convert_inferrence_output_to_evaluation_format.logging')
    def test_statistics_logging(self, mock_logging):
        """Test that statistics are properly logged."""
        input_data = [
            {'arxiv_id': '1234.5678',
                'predicted_authors': [{'name': 'Author'}]},
            {'arxiv_id': '2345.6789', 'predicted_authors': None},
            {'arxiv_id': '3456.7890', 'predicted_authors': []},
        ]

        with open(self.input_file, 'w') as f:
            for entry in input_data:
                f.write(json.dumps(entry) + '\n')
            f.write('invalid json\n')  # This will cause an error

        convert_content_to_predictions(
            str(self.input_file), str(self.output_file))

        info_calls = [call[0][0] for call in mock_logging.info.call_args_list]

        stats_found = False
        for msg in info_calls:
            if 'Total entries processed' in msg:
                stats_found = True
                break
        self.assertTrue(stats_found, "Statistics not found in log messages")


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_realistic_data(self):
        input_file = Path(self.temp_dir) / 'realistic.jsonl'
        output_file = Path(self.temp_dir) / 'realistic_output.json'

        realistic_data = [
            {
                "arxiv_id": "arXiv:2003.03151",
                "predicted_authors": [
                    {
                        "name": "Meiling Fang",
                        "affiliations": [
                            "Fraunhofer Institute for Computer Graphics Research IGD, Darmstadt, Germany",
                            "Mathematical and Applied Visual Computing, TU Darmstadt, Darmstadt, Germany"
                        ]
                    },
                    {
                        "name": "Naser Damer",
                        "affiliations": [
                            "Fraunhofer Institute for Computer Graphics Research IGD, Darmstadt, Germany",
                            "Mathematical and Applied Visual Computing, TU Darmstadt, Darmstadt, Germany"
                        ]
                    }
                ],
                "raw_output": "```json\\n[...]\\n```",
                "error": None,
                "processing_time": 10.65
            },
            {
                "arxiv_id": "arXiv:1709.02995",
                "predicted_authors": None,
                "raw_output": "```json\\n[...]\\n```",
                "error": "Failed to parse JSON",
                "processing_time": 3.85
            }
        ]

        with open(input_file, 'w') as f:
            for entry in realistic_data:
                f.write(json.dumps(entry) + '\n')

        convert_content_to_predictions(str(input_file), str(output_file))

        with open(output_file, 'r') as f:
            output = json.load(f)

        self.assertEqual(len(output), 2)
        self.assertIn('arXiv:2003.03151', output)
        self.assertIn('arXiv:1709.02995', output)

        # Check wrapped in predicted_authors
        self.assertEqual(len(output['arXiv:2003.03151']['predicted_authors']), 2)
        self.assertEqual(output['arXiv:2003.03151']['predicted_authors'][0]['name'], 'Meiling Fang')

        self.assertEqual(output['arXiv:1709.02995']['predicted_authors'], [])


if __name__ == '__main__':
    unittest.main()
