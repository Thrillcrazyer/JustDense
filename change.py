import csv
import re

def parse_to_csv_classification(input_path, output_path):
    rows = []

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by empty lines to handle entries as blocks
    entries = [entry.strip() for entry in content.strip().split('\n\n') if entry.strip()]
    
    for entry in entries:
        lines = entry.strip().split('\n')
        if not lines:
            continue
            
        name_line = lines[0].strip()
        
        # Find accuracy line
        accuracy = None
        for line in lines[1:]:
            if 'accuracy:' in line.lower():
                try:
                    accuracy = line.split(':')[1].strip()
                    break
                except IndexError:
                    continue
        
        # Parse name line
        parts = name_line.split('_')
        if len(parts) < 4:
            print(f"Skipping malformed entry: {name_line}")
            continue

        task_type = parts[0]
        dataset = parts[1]
        model = parts[2]
        benchmark = parts[3]

        rows.append([task_type, dataset, model, benchmark, accuracy])
        print(f"Parsed: {task_type}, {dataset}, {model}, {benchmark}, {accuracy}")

    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['task_type', 'dataset', 'model', 'benchmark', 'accuracy'])  # header
        writer.writerows(rows)
    
    print(f"Total entries processed: {len(rows)}")

def parse_to_csv_anomaly(input_path, output_path='result_anomaly_detection.csv'):
    """
    Reads anomaly detection results from a .txt file, parses them, and saves them to a CSV file.

    Parameters:
        input_path (str): Path to the input .txt file containing the result strings.
        output_path (str): Path to save the CSV file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 결과 블록을 두 줄 이상의 entry 단위로 나눔 (공백 줄 기준)
    entries = [entry.strip() for entry in content.strip().split('\n\n') if entry.strip()]
    parsed_data = []

    for entry in entries:
        try:
            # 2. 줄 분리 후 첫 줄은 모델 정보, 나머지 줄은 전부 이어서 메트릭 문자열로 처리
            lines = entry.strip().split('\n')
            line = lines[0].strip()
            metrics = " ".join(lines[1:]).strip()

            name_parts = line.split('_')
            task = name_parts[0] + "_" + name_parts[1]
            dataset = name_parts[2]
            model = name_parts[3]

            accuracy = re.search(r'Accuracy\s*:\s*([0-9.]+)', metrics).group(1)
            precision = re.search(r'Precision\s*:\s*([0-9.]+)', metrics).group(1)
            recall = re.search(r'Recall\s*:\s*([0-9.]+)', metrics).group(1)
            fscore = re.search(r'F-score\s*:\s*([0-9.]+)', metrics).group(1)

            parsed_data.append([task, dataset, model, accuracy, precision, recall, fscore])
        except Exception as e:
            print(f"Error parsing entry:\n{entry}\n{e}\n")

    # CSV 저장
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task', 'dataset', 'model', 'accuracy', 'precision', 'recall', 'fscore'])
        writer.writerows(parsed_data)

    print(f"CSV file saved to: {output_path}")

def parse_to_csv_imputation(input_path, output_path='imputation_results.csv'):
    """
    Reads imputation results from a .txt file, parses them, and saves them to a CSV file.

    Parameters:
        input_path (str): Path to the input .txt file containing the result strings.
        output_path (str): Path to save the CSV file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 빈 줄로 구분된 엔트리 블록을 나눔
    entries = [entry.strip() for entry in content.strip().split('\n\n') if entry.strip()]
    parsed_data = []

    for entry in entries:
        try:
            lines = entry.strip().split('\n')
            line = lines[0].strip()
            metrics = " ".join(lines[1:]).strip()
            if 'PatchTST' in line:
                name_parts = line.split('_')
                task = name_parts[0]  # 'imputation'
                dataset = name_parts[1]  # e.g., 'ETTh1'
                mask_ratio = name_parts[3]  # e.g., '0.125'
                model = name_parts[4]  # e.g., 'PatchTST'
            else:
                name_parts = line.split('_')
                task = name_parts[0]  # 'imputation'
                dataset = name_parts[1]  # e.g., 'ETTh1'
                mask_ratio = name_parts[3]  # e.g., '0.125'
                model = name_parts[4]  # e.g., 'Autoformer'

            mse = re.search(r'mse[:：]\s*([0-9.eE+-]+)', metrics).group(1)
            mae = re.search(r'mae[:：]\s*([0-9.eE+-]+)', metrics).group(1)

            parsed_data.append([task, dataset, mask_ratio, model, mse, mae])
        except Exception as e:
            print(f"Error parsing entry:\n{entry}\n{e}\n")

    # CSV 저장
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task', 'dataset', 'mask_ratio', 'model', 'mse', 'mae'])
        writer.writerows(parsed_data)

    print(f"CSV file saved to: {output_path}")  

def parse_to_csv_longterm_forecast(input_path, output_path='longterm_forecast_parsed.csv'):
    """
    result_long_term_forecast.txt에서 dataset, pred_len, model, mse, mae 추출하여 CSV로 저장
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = [entry.strip() for entry in content.strip().split('\n\n') if entry.strip()]
    parsed_data = []

    for entry in entries:
        try:
            lines = entry.split('\n')
            name_line = lines[0].strip()
            line = lines[0].strip()
            metrics_line = " ".join(lines[1:]).strip()
            name_parts = name_line.split('_')
            if 'ModernTCN' in line:
                dataset = name_parts[3]  # e.g., 'ECL'
                seq_len= name_parts[7]  # seq_len is not used in the output
                pred_len = name_parts[8]
                model = name_parts[4]
            else:
                # 예시: long_term_forecast_ETTm1_96_96_PatchTST_ETTm1_ftM_...
                # dataset: ETTm1, pred_len: 96, model: PatchTST
                dataset = name_parts[3]
                seq_len= name_parts[4]  # seq_len is not used in the output
                pred_len = name_parts[5]
                model = name_parts[6]

            mse = re.search(r'mse[:：]\s*([0-9.eE+-]+)', metrics_line).group(1)
            mae = re.search(r'mae[:：]\s*([0-9.eE+-]+)', metrics_line).group(1)

            parsed_data.append([dataset, seq_len,pred_len, model, mse, mae])
        except Exception as e:
            print(f"Error parsing entry:\n{entry}\n{e}\n")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset','seq_len','pred_len', 'model', 'mse', 'mae'])
        writer.writerows(parsed_data)

    print(f"CSV file saved to: {output_path}")

if __name__ == '__main__':
    # 사용 예시
    #parse_to_csv_classification('result_classification.txt','result_classification.csv')
    #parse_to_csv_imputation('result_imputation.txt')
    #parse_to_csv_anomaly('result_anomaly_detection.txt')
    parse_to_csv_longterm_forecast('result_long_term_forecast.txt')