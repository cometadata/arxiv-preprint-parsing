use std::{
    collections::HashMap,
    fs::{File, create_dir_all},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use csv::Writer as CsvWriter;
use flate2::read::GzDecoder;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, warn, LevelFilter};
use rayon::prelude::*;
use serde::Deserialize;
use simple_logger::SimpleLogger;
use time::macros::format_description;

mod memory_usage {
    #[cfg(target_os = "linux")]
    pub mod linux {
        use super::MemoryStats;
        use std::fs::read_to_string;
        pub fn get_memory_usage() -> Option<MemoryStats> {
            let pid = std::process::id();
            let status_file = format!("/proc/{}/status", pid);
            let content = read_to_string(status_file).ok()?;
            let mut vm_rss_kb = None;
            let mut vm_size_kb = None;
            for line in content.lines() {
                if line.starts_with("VmRSS:") {
                    vm_rss_kb = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<f64>().ok());
                } else if line.starts_with("VmSize:") {
                    vm_size_kb = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<f64>().ok());
                }
                if vm_rss_kb.is_some() && vm_size_kb.is_some() {
                    break;
                }
            }
            let rss_mb = vm_rss_kb.map(|kb| kb / 1024.0);
            let vm_size_mb = vm_size_kb.map(|kb| kb / 1024.0);
            let mut percent = None;
            if let Ok(meminfo) = read_to_string("/proc/meminfo") {
                if let Some(mem_total_kb) = meminfo
                    .lines()
                    .find(|line| line.starts_with("MemTotal:"))
                    .and_then(|line| line.split_whitespace().nth(1))
                    .and_then(|s| s.parse::<f64>().ok())
                {
                    if mem_total_kb > 0.0 {
                        if let Some(rss) = vm_rss_kb {
                            percent = Some((rss / mem_total_kb) * 100.0);
                        }
                    }
                }
            }
            Some(MemoryStats {
                rss_mb: rss_mb.unwrap_or(0.0),
                vm_size_mb: vm_size_mb.unwrap_or(0.0),
                percent,
            })
        }
    }
    #[cfg(target_os = "macos")]
    pub mod macos {
        use super::MemoryStats;
        use std::process::Command;
        pub fn get_memory_usage() -> Option<MemoryStats> {
            let pid = std::process::id();
            let ps_output_rss = Command::new("ps")
                .args(&["-o", "rss=", "-p", &pid.to_string()])
                .output()
                .ok()?;
            let rss_kb = String::from_utf8_lossy(&ps_output_rss.stdout)
                .trim()
                .parse::<f64>()
                .ok()?;
            let ps_output_vsz = Command::new("ps")
                .args(&["-o", "vsz=", "-p", &pid.to_string()])
                .output()
                .ok()?;
            let vsz_kb = String::from_utf8_lossy(&ps_output_vsz.stdout)
                .trim()
                .parse::<f64>()
                .ok()?;
            let rss_mb = rss_kb / 1024.0;
            let vm_size_mb = vsz_kb / 1024.0;
            let mut percent = None;
            if let Ok(hw_mem_output) = Command::new("sysctl").args(&["-n", "hw.memsize"]).output() {
                if let Ok(total_bytes_str) = String::from_utf8(hw_mem_output.stdout) {
                    if let Ok(total_bytes) = total_bytes_str.trim().parse::<f64>() {
                        let total_kb = total_bytes / 1024.0;
                        if total_kb > 0.0 {
                            percent = Some((rss_kb / total_kb) * 100.0);
                        }
                    }
                }
            }
            Some(MemoryStats {
                rss_mb,
                vm_size_mb,
                percent,
            })
        }
    }
    #[cfg(target_os = "windows")]
    pub mod windows {
        use super::MemoryStats;
        use std::process::Command;
        pub fn get_memory_usage() -> Option<MemoryStats> {
            let pid = std::process::id();
            let wmic_output = Command::new("wmic")
                .args(&[
                    "process",
                    "where",
                    &format!("ProcessId={}", pid),
                    "get",
                    "WorkingSetSize,",
                    "PageFileUsage",
                    "/value",
                ])
                .output()
                .ok()?;
            let output_str = String::from_utf8_lossy(&wmic_output.stdout);
            let mut rss_bytes: Option<f64> = None;
            let mut vm_kb: Option<f64> = None;
            for line in output_str.lines() {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let key = parts[0].trim();
                    let value = parts[1].trim();
                    match key {
                        "PageFileUsage" => vm_kb = value.parse::<f64>().ok(),
                        "WorkingSetSize" => rss_bytes = value.parse::<f64>().ok(),
                        _ => {}
                    }
                }
            }
            let rss_mb = rss_bytes.map(|b| b / (1024.0 * 1024.0));
            let vm_size_mb = vm_kb.map(|kb| kb / 1024.0);
            let mut percent = None;
            if let Ok(mem_output) = Command::new("wmic")
                .args(&["ComputerSystem", "get", "TotalPhysicalMemory", "/value"])
                .output()
            {
                let mem_str = String::from_utf8_lossy(&mem_output.stdout);
                if let Some(total_bytes_str) = mem_str
                    .lines()
                    .find(|line| line.starts_with("TotalPhysicalMemory="))
                    .and_then(|line| line.split('=').nth(1))
                {
                    if let Ok(total_bytes) = total_bytes_str.trim().parse::<f64>() {
                        if total_bytes > 0.0 {
                            if let Some(rss) = rss_bytes {
                                percent = Some((rss / total_bytes) * 100.0);
                            }
                        }
                    }
                }
            }
            Some(MemoryStats {
                rss_mb: rss_mb.unwrap_or(0.0),
                vm_size_mb: vm_size_mb.unwrap_or(0.0),
                percent,
            })
        }
    }
    #[derive(Debug)]
    pub struct MemoryStats {
        pub rss_mb: f64,
        pub vm_size_mb: f64,
        pub percent: Option<f64>,
    }
    #[cfg(target_os = "linux")]
    use self::linux::get_memory_usage;
    #[cfg(target_os = "macos")]
    use self::macos::get_memory_usage;
    #[cfg(target_os = "windows")]
    use self::windows::get_memory_usage;
    #[cfg(not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows"
    )))]
    pub fn get_memory_usage() -> Option<MemoryStats> {
        None
    }
    pub fn log_memory_usage(note: &str) {
        use log::info; 
        if let Some(stats) = get_memory_usage() {
            let percent_str = stats
                .percent
                .map_or_else(|| "N/A".to_string(), |p| format!("{:.1}%", p));
            let vm_str = if stats.vm_size_mb > 0.0 {
                format!("{:.1} MB virtual/commit", stats.vm_size_mb)
            } else {
                "N/A virtual".to_string()
            };
            info!(
                "Memory usage ({}): {:.1} MB physical (RSS), {}, {} of system memory",
                note, stats.rss_mb, vm_str, percent_str
            );
        } else {
            info!(
                "Memory usage tracking not available or failed on this platform ({})",
                std::env::consts::OS
            );
        }
    }
}

#[derive(Parser)]
#[command(name = "OpenAlex Source and Affiliation Analyzer")]
#[command(about = "Analyzes author affiliation completeness for works from a specific OpenAlex source ID and outputs categorized work IDs/DOIs to CSVs.")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, help = "Directory containing input JSONL.gz files", required = true)]
    input_dir: String,

    #[arg(short, long, help = "Output directory for summary log file and CSVs", required = true)]
    output_dir: String,

    #[arg(short = 's', long, help = "OpenAlex Source ID to analyze (e.g., S4306400194)", required = true)]
    source_id: String,

    #[arg(short, long, default_value = "INFO", help = "Logging level (DEBUG, INFO, WARN, ERROR)")]
    log_level: String,

    #[arg(short = 't', long, default_value = "0", help = "Number of threads to use (0 for auto)")]
    threads: usize,

    #[arg(long, default_value = "60", help = "Interval in seconds to log statistics")]
    stats_interval: u64,
}

#[derive(Deserialize, Debug)]
struct OpenAlexRecord {
    id: Option<String>, 
    doi: Option<String>, 
    primary_location: Option<PrimaryLocation>,
    authorships: Option<Vec<Authorship>>,
}

#[derive(Deserialize, Debug)]
struct PrimaryLocation {
    source: Option<Source>,
}

#[derive(Deserialize, Debug)]
struct Source {
    id: Option<String>,
}

#[derive(Deserialize, Debug)]
struct Authorship {
    institutions: Option<Vec<Institution>>,
}

#[derive(Deserialize, Debug)]
struct Institution {}

#[derive(Debug)]
struct WorkIdentifier {
    openalex_id: String,
    doi: Option<String>,
}

struct CategorizedCsvWriterManager {
    output_dir: PathBuf,
    source_id: String,
    writers: Mutex<HashMap<String, CsvWriter<File>>>,
}

static ALL_AFF_KEY: &str = "all_affiliations";
static PARTIAL_AFF_KEY: &str = "partial_affiliations";
static NO_AFF_KEY: &str = "no_affiliations";
static NO_AUTH_KEY: &str = "no_authors";


impl CategorizedCsvWriterManager {
    fn new(output_dir: PathBuf, source_id: String) -> Self {
        Self {
            output_dir,
            source_id,
            writers: Mutex::new(HashMap::new()),
        }
    }

    fn write_record(&self, category_key: &str, work_identifier: &WorkIdentifier) -> Result<()> {
        let mut writers_guard = self.writers.lock().unwrap();
        
        if !writers_guard.contains_key(category_key) {
            let file_name = format!("source_{}_{}.csv", self.source_id, category_key);
            let file_path = self.output_dir.join(file_name);
            
            let file = File::create(&file_path)
                .with_context(|| format!("Failed to create CSV file: {}", file_path.display()))?;
            let mut writer = CsvWriter::from_writer(file);
            
            writer.write_record(&["openalex_id", "doi"])
                .with_context(|| format!("Failed to write CSV header to {}", file_path.display()))?;
            
            writers_guard.insert(category_key.to_string(), writer);
            debug!("Opened CSV writer for category '{}': {}", category_key, file_path.display());
        }

        let writer = writers_guard.get_mut(category_key)
            .ok_or_else(|| anyhow!("CSV writer for category '{}' unexpectedly missing after check/insert.", category_key))?;
        
        let doi_str = work_identifier.doi.as_deref().unwrap_or("");
        writer.write_record(&[&work_identifier.openalex_id, doi_str])
            .with_context(|| format!("Failed to write CSV record (ID: {}) to category {}", work_identifier.openalex_id, category_key))?;
        
        Ok(())
    }

    fn close_all_writers(&self) -> Result<()> {
        info!("Closing all categorized CSV writers...");
        let mut writers_guard = self.writers.lock().unwrap();
        let mut errors: Vec<String> = Vec::new();

        for (category, mut writer) in writers_guard.drain() {
            if let Err(e) = writer.flush() {
                let err_msg = format!("Failed to flush CSV writer for category {}: {}", category, e);
                error!("{}", err_msg);
                errors.push(err_msg);
            }
            info!("Closed CSV writer for category '{}'.", category);
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(anyhow!("Errors occurred while closing CSV writers:\n - {}", errors.join("\n - ")))
        }
    }
}

#[derive(Default)]
struct Stats {
    lines_read: AtomicU64,
    json_parse_errors: AtomicU64,
    source_match_errors: AtomicU64,
    works_found_for_source: AtomicU64,
    works_with_all_affiliations: AtomicU64,
    works_with_partial_affiliations: AtomicU64,
    works_with_no_affiliations: AtomicU64,
    works_with_no_authors: AtomicU64,
}

impl Stats {
    fn new() -> Self {
        Default::default()
    }

    fn log_current_stats(&self, stage: &str) {
        let lines_read = self.lines_read.load(Ordering::Relaxed);
        let json_err = self.json_parse_errors.load(Ordering::Relaxed);
        let match_err = self.source_match_errors.load(Ordering::Relaxed);
        let works_found = self.works_found_for_source.load(Ordering::Relaxed);
        let works_all_aff = self.works_with_all_affiliations.load(Ordering::Relaxed);
        let works_partial_aff = self.works_with_partial_affiliations.load(Ordering::Relaxed);
        let works_no_aff = self.works_with_no_affiliations.load(Ordering::Relaxed);
        let works_no_auth = self.works_with_no_authors.load(Ordering::Relaxed);

        info!("--- Periodic Stats ({}) ---", stage);
        info!(" Lines Read (Input): {}", lines_read);
        info!(" JSON Parse Errors: {}", json_err);
        info!(" Source ID Match Errors: {}", match_err);
        info!(" Works found for target source: {}", works_found);
        info!("    ↳ Fully Affiliated: {}", works_all_aff);
        info!("    ↳ Partially Affiliated: {}", works_partial_aff);
        info!("    ↳ Zero Affiliations (Authors Present): {}", works_no_aff);
        info!("    ↳ No Authors Listed: {}", works_no_auth);
        info!("------------------------------");
    }
}

fn find_gz_files<P: AsRef<Path>>(directory: P) -> Result<Vec<PathBuf>> {
    let pattern = directory.as_ref().join("**/*.gz");
    let pattern_str = pattern.to_string_lossy();
    info!("Searching for files matching pattern: {}", pattern_str);

    let paths: Vec<PathBuf> = glob(&pattern_str)?
        .filter_map(Result::ok) 
        .filter(|path| path.is_file()) 
        .filter(|path| { 
            path.file_name()
                .and_then(|name| name.to_str())
                .map_or(false, |name_str| !name_str.ends_with(".csv.gz"))
        })
        .collect();

    if paths.is_empty() {
        warn!(
            "No .gz files found (excluding .csv.gz) matching the pattern: {}",
            pattern_str
        );
    }
    Ok(paths)
}


fn has_affiliation(author: &Authorship) -> bool {
    match &author.institutions {
        Some(inst_vec) => !inst_vec.is_empty(),
        None => false,
    }
}

fn format_elapsed(elapsed: Duration) -> String {
    let total_secs = elapsed.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s {}ms", seconds, elapsed.subsec_millis())
    }
}

fn calculate_median(numbers: &mut Vec<f64>) -> Option<f64> {
    if numbers.is_empty() {
        return None;
    }
    numbers.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = numbers.len();
    let mid = len / 2;
    if len % 2 == 0 {
        Some((numbers[mid - 1] + numbers[mid]) / 2.0)
    } else {
        Some(numbers[mid])
    }
}

fn main() -> Result<()> {
    let main_start_time = Instant::now();
    let cli = Cli::parse();

    let log_level = match cli.log_level.to_uppercase().as_str() {
        "DEBUG" => LevelFilter::Debug,
        "INFO" => LevelFilter::Info,
        "WARN" | "WARNING" => LevelFilter::Warn,
        "ERROR" => LevelFilter::Error,
        _ => {
            eprintln!("Invalid log level '{}', defaulting to INFO.", cli.log_level);
            LevelFilter::Info
        }
    };
    SimpleLogger::new()
        .with_level(log_level)
        .with_timestamp_format(format_description!(
            "[year]-[month]-[day] [hour]:[minute]:[second]"
        ))
        .init()?;

    info!("Starting OpenAlex Source Affiliation Analyzer v{}", env!("CARGO_PKG_VERSION")); 
    memory_usage::log_memory_usage("initial");

    let output_dir = PathBuf::from(&cli.output_dir);
    create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create output directory: {}", output_dir.display()))?;
    
    let summary_file_path = output_dir.join(format!("summary_text_{}.txt", cli.source_id));

    let csv_manager = Arc::new(CategorizedCsvWriterManager::new(
        output_dir.clone(),
        cli.source_id.clone(),
    ));


    let num_threads = if cli.threads == 0 {
        let cores = num_cpus::get();
        info!("Auto-detected {} CPU cores. Using {} threads.", cores, cores);
        cores
    } else {
        info!("Using specified {} threads.", cli.threads);
        cli.threads
    };
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
    {
        error!("Failed to build global thread pool: {}. Proceeding with default.", e);
    }

    info!("Searching for input files in: {}", cli.input_dir);
    let files = find_gz_files(&cli.input_dir)?; 
    if files.is_empty() {
        warn!("No suitable .gz (JSONL) files found to process in {}. Exiting.", cli.input_dir);
        return Ok(());
    }
    info!("Found {} input .gz (JSONL) files to process.", files.len());

    info!("Target Source ID: {}", cli.source_id);
    info!("Output directory for summary & CSVs: {}", output_dir.display());
    info!("Statistics logging interval: {} seconds.", cli.stats_interval);

    let stats = Arc::new(Stats::new());
    let target_source_id = Arc::new(cli.source_id.clone());

    let stats_thread_running = Arc::new(Mutex::new(true));
    let stats_interval_duration = Duration::from_secs(cli.stats_interval);
    let stats_clone_for_thread = Arc::clone(&stats);
    let stats_thread_running_clone = Arc::clone(&stats_thread_running);
    let stats_thread = thread::spawn(move || {
        info!("Stats logging thread started.");
        let mut last_log_time = Instant::now();
        loop {
            match stats_thread_running_clone.try_lock() {
                Ok(guard) => if !*guard { info!("Stats thread received stop signal."); break; },
                Err(std::sync::TryLockError::WouldBlock) => { /* continue */ },
                Err(e) => { error!("Stats thread lock error: {}", e); break; }
            }
            thread::sleep(Duration::from_millis(500));
            if last_log_time.elapsed() >= stats_interval_duration {
                memory_usage::log_memory_usage("periodic check");
                stats_clone_for_thread.log_current_stats("Processing");
                last_log_time = Instant::now();
            }
        }
        info!("Stats logging thread finished.");
    });

    info!("--- Starting Processing ---");
    let processing_start_time = Instant::now();

    let progress_bar = ProgressBar::new(files.len() as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} @ {per_sec}) {msg}")
            .expect("Failed to create progress bar template")
            .progress_chars("=> "),
    );
    progress_bar.set_message("Processing: Starting...");

    let processing_results: Vec<Result<Vec<f64>, String>> = files
        .par_iter()
        .map(|filepath| {
            let pb_clone = progress_bar.clone();
            let stats_clone = Arc::clone(&stats);
            let target_id_clone = Arc::clone(&target_source_id);
            let csv_manager_clone = Arc::clone(&csv_manager);

            let mut local_percentages: Vec<f64> = Vec::new();
            let mut file_errors: Vec<String> = Vec::new();

            let file = match File::open(filepath) {
                Ok(f) => f,
                Err(e) => return Err(format!("Failed to open {}: {}", filepath.display(), e)),
            };
            let decoder = GzDecoder::new(file);
            let mut reader = BufReader::new(decoder);
            let mut byte_buffer = Vec::with_capacity(8192);

            loop {
                byte_buffer.clear();
                match reader.read_until(b'\n', &mut byte_buffer) {
                    Ok(0) => break, 
                    Ok(bytes_read) => {
                        if bytes_read == 0 { break; } 
                        stats_clone.lines_read.fetch_add(1, Ordering::Relaxed);
                        if byte_buffer.iter().all(|&b| b.is_ascii_whitespace()) {
                            continue; 
                        }

                        match serde_json::from_slice::<OpenAlexRecord>(&byte_buffer) {
                            Ok(record) => {
                                let mut source_id_matched_this_record = false;
                                let work_openalex_id_opt = record.id.clone(); 
                                let work_doi_opt = record.doi.clone();

                                if let Some(loc) = &record.primary_location {
                                    if let Some(src) = &loc.source {
                                        if let Some(id_str) = &src.id {
                                            let id_part = id_str.split('/').last().unwrap_or(id_str);
                                            if id_part == target_id_clone.as_str() {
                                                source_id_matched_this_record = true;
                                                stats_clone.works_found_for_source.fetch_add(1, Ordering::Relaxed);
                                                
                                                if let Some(work_openalex_id_val) = work_openalex_id_opt.as_ref() {
                                                    let work_id_for_csv = WorkIdentifier {
                                                        openalex_id: work_openalex_id_val.clone(),
                                                        doi: work_doi_opt.clone(),
                                                    };

                                                    let category_key: &str;
                                                    match &record.authorships {
                                                        Some(authors) if !authors.is_empty() => {
                                                            let total_authors = authors.len();
                                                            let affiliated_authors = authors.iter().filter(|a| has_affiliation(a)).count();
                                                            
                                                            if affiliated_authors == total_authors {
                                                                stats_clone.works_with_all_affiliations.fetch_add(1, Ordering::Relaxed);
                                                                category_key = ALL_AFF_KEY;
                                                            } else if affiliated_authors > 0 {
                                                                stats_clone.works_with_partial_affiliations.fetch_add(1, Ordering::Relaxed);
                                                                category_key = PARTIAL_AFF_KEY;
                                                            } else { 
                                                                stats_clone.works_with_no_affiliations.fetch_add(1, Ordering::Relaxed);
                                                                category_key = NO_AFF_KEY;
                                                            }
                                                            let percentage = (affiliated_authors as f64 / total_authors as f64) * 100.0;
                                                            local_percentages.push(percentage);
                                                        }
                                                        _ => { 
                                                            stats_clone.works_with_no_authors.fetch_add(1, Ordering::Relaxed);
                                                            category_key = NO_AUTH_KEY;
                                                        }
                                                    }
                                                    if let Err(e) = csv_manager_clone.write_record(category_key, &work_id_for_csv) {
                                                        let err_msg = format!("Failed to write record {} to CSV category {}: {}", work_id_for_csv.openalex_id, category_key, e);
                                                        error!("{}", err_msg); 
                                                        file_errors.push(err_msg); 
                                                    }
                                                } else if source_id_matched_this_record { 
                                                    warn!("Record matched source ID criteria but is missing its own OpenAlex ID. DOI: {:?}. Record will not be added to CSVs.", work_doi_opt);
                                                }
                                            }
                                        }
                                    }
                                }
                                if !source_id_matched_this_record && record.primary_location.is_some() {
                                    if record.primary_location.as_ref().and_then(|l| l.source.as_ref()).is_some()
                                        && record.primary_location.as_ref().and_then(|l| l.source.as_ref().and_then(|s| s.id.as_ref())).is_none() {
                                        stats_clone.source_match_errors.fetch_add(1, Ordering::Relaxed);
                                        debug!("Missing source ID field in record with OpenAlex ID: {:?}", work_openalex_id_opt);
                                    }
                                }
                            }
                            Err(e) => {
                                stats_clone.json_parse_errors.fetch_add(1, Ordering::Relaxed);
                                if stats_clone.json_parse_errors.load(Ordering::Relaxed) % 5000 == 1 {
                                    let snippet = String::from_utf8_lossy(&byte_buffer).chars().take(150).collect::<String>();
                                    warn!("JSON parse error in {}: {} (Line starts: '{}...')", filepath.display(), e, snippet);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let msg = format!("Read error in {}: {}", filepath.display(), e);
                        error!("{}", msg);
                        file_errors.push(msg);
                        break; 
                    }
                }
            } 

            pb_clone.inc(1);
            let file_name_msg = filepath.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| filepath.display().to_string());
            pb_clone.set_message(format!("Processed: {}", file_name_msg));

            if file_errors.is_empty() {
                Ok(local_percentages) 
            } else {
                Err(format!("Errors processing {}:\n - {}", filepath.display(), file_errors.join("\n - ")))
            }
        })
        .collect();

    progress_bar.finish_with_message("Processing complete.");
    let processing_duration = processing_start_time.elapsed();

    if let Err(e) = csv_manager.close_all_writers() {
        error!("Error(s) closing CSV writers: {}", e);
    } else {
        info!("All categorized CSV writers closed successfully.");
    }


    info!("Aggregating results for median calculation...");
    let mut all_percentages: Vec<f64> = Vec::new();
    let mut final_errors: Vec<String> = Vec::new(); 

    for result in processing_results { 
        match result {
            Ok(thread_percentages) => all_percentages.extend(thread_percentages),
            Err(e_str) => final_errors.push(e_str), 
        }
    }
    info!("Aggregation complete. Collected {} percentages for median calculation.", all_percentages.len());

    info!("--- Processing Summary ---");
    info!("Duration: {}", format_elapsed(processing_duration));

    if !final_errors.is_empty() {
        warn!("{} files encountered errors during processing (details logged per file).", final_errors.len());
        for error_msg in final_errors.iter().take(5) { 
             warn!("  File processing error summary: {}", error_msg);
        }
        if final_errors.len() > 5 {
            warn!("  ... and {} more file processing error summaries not shown here.", final_errors.len() - 5);
        }
    } else {
        info!("All files processed successfully (no file-level read/processing errors reported to main thread).");
    }

    info!("Calculating final statistics...");
    let total_works = stats.works_found_for_source.load(Ordering::Relaxed);
    let all_aff = stats.works_with_all_affiliations.load(Ordering::Relaxed);
    let partial_aff = stats.works_with_partial_affiliations.load(Ordering::Relaxed);
    let no_aff = stats.works_with_no_affiliations.load(Ordering::Relaxed);
    let no_auth = stats.works_with_no_authors.load(Ordering::Relaxed);

    let perc_all_aff = if total_works > 0 { (all_aff as f64 / total_works as f64) * 100.0 } else { 0.0 };
    let perc_partial_aff = if total_works > 0 { (partial_aff as f64 / total_works as f64) * 100.0 } else { 0.0 };
    let median_percentage = calculate_median(&mut all_percentages);

    stats.log_current_stats("Final Calculation");
    info!("-------------------- FINAL SUMMARY (Source ID: {}) --------------------", cli.source_id);
    info!(" Total execution time: {}", format_elapsed(main_start_time.elapsed()));
    info!(" Input files processed: {}", files.len()); 
    info!(" File processing/read errors reported to main thread: {}", final_errors.len());
    info!(" Total lines read from input files: {}", stats.lines_read.load(Ordering::Relaxed));
    info!(" JSON parsing errors: {}", stats.json_parse_errors.load(Ordering::Relaxed));
    info!(" --- Affiliation Stats for Source {} (Records Identified) ---", cli.source_id);
    info!(" Total works found for this source: {}", total_works);
    info!("   - Works with ALL authors affiliated: {} ({:.2}%)", all_aff, perc_all_aff);
    info!("   - Works with PARTIAL authors affiliated: {} ({:.2}%)", partial_aff, perc_partial_aff);
    info!("   - Works with NO authors affiliated (authors present): {}", no_aff);
    info!("   - Works with NO authors listed: {}", no_auth);
    match median_percentage {
        Some(median) => info!(" Median percentage of affiliated authors per work (based on {} works with authors): {:.2}%", all_percentages.len(), median),
        None => info!(" Median percentage of affiliated authors per work: N/A (No works with authors found or all had zero authors?)"),
    }
    info!(" --- CSV Output Summary (Records Identified for each category) ---");
    info!("   Identified for 'all_affiliations' CSV: {}", all_aff);
    info!("   Identified for 'partial_affiliations' CSV: {}", partial_aff);
    info!("   Identified for 'no_affiliations' CSV: {}", no_aff);
    info!("   Identified for 'no_authors' CSV: {}", no_auth);
    memory_usage::log_memory_usage("final");
    info!("--------------------------------------------------------------------");

    info!("Writing summary text to {}", summary_file_path.display());
    {
        let mut writer = File::create(&summary_file_path) 
            .with_context(|| format!("Failed to create summary file: {}", summary_file_path.display()))?;
        writeln!(writer, "--- OpenAlex Source Affiliation Analysis Summary ---")?;
        writeln!(writer, "Timestamp: {}", time::OffsetDateTime::now_utc().to_string())?;
        writeln!(writer, "Source ID Analyzed: {}", cli.source_id)?;
        writeln!(writer, "Input Directory: {}", cli.input_dir)?;
        writeln!(writer, "Output Directory: {}", cli.output_dir)?;
        writeln!(writer, "Total Execution Time: {}", format_elapsed(main_start_time.elapsed()))?;
        writeln!(writer, "Input Files Processed: {} ({} reported errors to main thread)", files.len(), final_errors.len())?;
        writeln!(writer, "Total Lines Read: {}", stats.lines_read.load(Ordering::Relaxed))?;
        writeln!(writer, "JSON Parse Errors: {}", stats.json_parse_errors.load(Ordering::Relaxed))?;
        writeln!(writer, "--- Affiliation Statistics (Records Identified) ---")?;
        writeln!(writer, "Total Works Found: {}", total_works)?;
        writeln!(writer, "  Fully Affiliated: {} ({:.2}%)", all_aff, perc_all_aff)?;
        writeln!(writer, "  Partially Affiliated: {} ({:.2}%)", partial_aff, perc_partial_aff)?;
        writeln!(writer, "  Zero Affiliations (Authors Present): {}", no_aff)?;
        writeln!(writer, "  No Authors Listed: {}", no_auth)?;
        match median_percentage {
            Some(median) => writeln!(writer, "Median Affiliation Percentage per Work (based on {} works with authors): {:.2}%", all_percentages.len(), median)?,
            None => writeln!(writer, "Median Affiliation Percentage per Work: N/A")?,
        }
        writeln!(writer, "--- CSV Output (Records Identified for each category) ---")?;
        writeln!(writer, "  Identified for All Affiliations CSV: {}", all_aff)?;
        writeln!(writer, "  Identified for Partial Affiliations CSV: {}", partial_aff)?;
        writeln!(writer, "  Identified for No Affiliations CSV: {}", no_aff)?;
        writeln!(writer, "  Identified for No Authors CSV: {}", no_auth)?;
        writeln!(writer, "--- End Summary ---")?;
    }

    info!("Signaling stats thread to stop...");
    if let Ok(mut running_guard) = stats_thread_running.lock() {
        *running_guard = false;
        drop(running_guard);
        info!("Waiting for stats thread to finish...");
        if let Err(e) = stats_thread.join() {
            error!("Error joining stats thread: {:?}", e);
        } else {
            info!("Stats thread joined successfully.");
        }
    } else {
        error!("Failed to lock stats thread running flag to signal stop.");
    }

    if !final_errors.is_empty() { 
        Err(anyhow!("Processing finished with file processing errors reported to the main thread."))
    } else {
        Ok(())
    }
}
