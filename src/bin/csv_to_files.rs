use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use csv::ReaderBuilder;

/// Format the article number with leading zeros (article_001, article_002, etc.)
fn format_article_number(idx: usize) -> String {
    format!("article_{:03}", idx + 1)
}

/// Simple utility to extract articles from CSV to individual files
fn main() -> Result<(), Box<dyn Error>> {
    // Input CSV file path
    let csv_path = "90minFootballTransferNewsNLP.csv";
    
    // Output directory
    let out_dir = "articles";
    
    // Ensure output directory exists
    fs::create_dir_all(out_dir)?;
    
    println!("Reading articles from: {}", csv_path);
    
    // Configure CSV reader - handle quoted fields correctly
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .double_quote(true)
        .from_path(csv_path)?;
    
    let mut article_count = 0;
    
    // Process each record in the CSV
    for result in reader.records() {
        let record = result?;
        
        // Extract fields
        if record.len() < 4 {
            println!("Skipping record with insufficient fields");
            continue;
        }
        
        let title = &record[0];
        let date = &record[1];
        let link = &record[2];
        let content = &record[3];
        
        // Create file name for this article
        let file_name = format!("{}/{}.txt", out_dir, format_article_number(article_count));
        let file_path = Path::new(&file_name);
        
        // Create and write to the file
        let mut file = File::create(&file_path)?;
        
        // Write article data to file
        writeln!(file, "{}", title)?;  // Title on first line
        writeln!(file, "{}", date)?;   // Date on second line
        writeln!(file, "{}", link)?;   // Link on third line
        writeln!(file, "{}", content)?; // Content on subsequent lines
        
        article_count += 1;
    }
    
    println!("Successfully extracted {} articles to the '{}' directory", article_count, out_dir);
    
    Ok(())
}