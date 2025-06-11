from utils.m4_summary import M4Summary
import os

def get_folder_names_in_directory(target_directory):
    """
    특정 폴더 내의 폴더들 이름을 반환하는 함수

    Args:
        target_directory (str): 검색할 대상 디렉토리 경로

    Returns:
        list: 폴더 이름 리스트
    """
    folder_names = []
    try:
        if not os.path.exists(target_directory):
            print(f"디렉토리가 존재하지 않습니다: {target_directory}")
            return folder_names

        for item in os.listdir(target_directory):
            item_path = os.path.join(target_directory, item)
            if os.path.isdir(item_path):
                folder_names.append(item)
    except PermissionError:
        print(f"디렉토리에 접근할 권한이 없습니다: {target_directory}")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

    return folder_names

if __name__ == "__main__":
    all_folders = get_folder_names_in_directory('/workspace/m4_results')
    for folder in all_folders:
        print(f"Found folder: {folder}")
        m4_summary = M4Summary(file_path=folder, root_path='/workspace/m4_results')
        smape_results, owa_results, mape, mase = m4_summary.evaluate()
        print('smape:', smape_results)
        print('mape:', mape)
        print('mase:', mase)
        print('owa:', owa_results)
        print(f"Evaluation completed for folder: {folder}")