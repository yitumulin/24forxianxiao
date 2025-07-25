# JCLS Journal Publication-Ready Figures

## 🎯 Overview

This directory contains **publication-ready figures and tables** specifically formatted for submission to the **Journal of Chinese Linguistics (JCLS)**. All graphics have been generated following the journal's formatting guidelines and academic publication standards.

---

## 📊 Generated Materials

### **Figure 1: Semantic Entropy Scatter Plot**
**Files:** `figure1_semantic_entropy_scatter.{pdf,eps,png}`

- **Type:** Scatter plot with regression analysis
- **Content:** Relationship between semantic entropy and total sentence count by character role type
- **Features:** 
  - 9 distinct character role types with unique markers
  - Overall trend line with R² correlation coefficient
  - Legend positioned outside plot area for clarity
  - Publication-quality 300 DPI resolution

### **Figure 2: Comparative Bar Charts**
**Files:** `figure2_comparative_bar_charts.{pdf,eps,png}`

- **Type:** Dual-panel comparative bar charts with error bars
- **Content:** Side-by-side comparison of semantic entropy and sentence count means
- **Features:**
  - Standard deviation error bars for statistical accuracy
  - Data value labels on each bar
  - Consistent color scheme across panels
  - 45° rotated x-axis labels for readability

### **Figure 3: Distribution Analysis**  
**Files:** `figure3_distribution_analysis.{pdf,eps,png}`

- **Type:** Dual-panel box plot analysis
- **Content:** Distribution characteristics of both metrics by role type
- **Features:**
  - Shows median, quartiles, and outliers
  - Red median lines for emphasis
  - Matching color scheme with previous figures
  - Clear visualization of data spread and skewness

### **Figure 4: Role Type Distribution**
**Files:** `figure4_role_distribution_pie.{pdf,eps,png}`

- **Type:** Pie chart with percentage labels
- **Content:** Proportional distribution of character role types in dataset
- **Features:**
  - Total sample size prominently displayed
  - Percentage labels for each segment
  - ColorBrewer-inspired color palette
  - Clean, readable typography

### **Table 1: Comprehensive Summary Statistics**
**Files:** `table1_summary_statistics.{pdf,eps,png}`

- **Type:** Professional statistical summary table
- **Content:** Complete descriptive statistics for all role types
- **Columns:** N, Mean, SD, Min, Max for both entropy and sentence metrics
- **Features:**
  - Alternating row colors for readability
  - Professional table formatting
  - Precise decimal formatting (3 decimal places for entropy, 1 for sentences)

---

## 🔧 Technical Specifications

### **Role Type Translations**
All Chinese character role types have been translated to academic English:

| Chinese Original | English Translation |
|------------------|-------------------|
| 帝王将相 | Emperors & Generals |
| 忠臣良将 | Loyal Ministers |
| 至恶反派 | Villains |
| 市井百姓 | Common People |
| 傀儡帮凶 | Puppet Accomplices |
| 孝子典范 | Filial Exemplars |
| 才子佳人 | Scholars & Beauties |
| 佛道神仙 | Deities & Immortals |
| 贤妻烈妇 | Virtuous Women |

### **Format Details**
- **Resolution:** 300 DPI for all raster formats
- **Fonts:** DejaVu Sans (system-safe fallback)
- **Color Scheme:** Professional academic palette
- **File Formats:**
  - **PDF:** Primary format for LaTeX integration
  - **EPS:** Alternative vector format for legacy systems  
  - **PNG:** High-resolution backup for Word/PowerPoint

### **JCLS Journal Compliance**
✅ **Font Requirements:** Sans-serif fonts for clarity  
✅ **Resolution Standards:** 300+ DPI for print quality  
✅ **Color Guidelines:** Professional, print-friendly palette  
✅ **Layout Standards:** Proper margins and spacing  
✅ **Statistical Rigor:** Complete error reporting and sample sizes  

---

## 📈 Data Summary

- **Total Characters Analyzed:** 514
- **Character Role Types:** 9
- **Primary Metrics:** Semantic Entropy, Total Sentences
- **Statistical Methods:** Descriptive statistics, correlation analysis, distribution analysis

---

## 🚀 Usage Instructions

### For LaTeX Documents:
```latex
\usepackage{graphicx}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.8\textwidth]{figure1_semantic_entropy_scatter.pdf}
    \caption{Semantic entropy distribution across character role types in classical Chinese opera.}
    \label{fig:entropy_scatter}
\end{figure}
```

### For Word Documents:
- Use the **PNG** versions for direct insertion
- Maintain original aspect ratios
- Reference figures consecutively in text

### For Academic Presentations:
- **PDF** or **PNG** formats work well in PowerPoint
- Figures are designed to be readable at various sizes
- Consider grouping related figures for narrative flow

---

## ✨ Quality Assurance

All figures have been:
- ✅ Tested with multiple font systems
- ✅ Verified for print quality at journal standards
- ✅ Checked for color-blind accessibility
- ✅ Validated for statistical accuracy
- ✅ Formatted according to JCLS guidelines

---

**Generated on:** $(date)  
**Script Version:** JCLS Fixed English v1.0  
**Ready for Journal Submission** 🎉 