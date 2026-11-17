"""Resume frozen inference with only the corrected report exporter installed."""
import negative_search as original
from negative_report import report
original.report=report
if __name__=="__main__":original.main()
