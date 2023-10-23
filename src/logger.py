from pathlib import Path
import shutil
import os
from datetime import datetime

class print_logger :
    def __init__(self, raw_log_path, log_name : str) : 
        log_path = Path(raw_log_path)
        log_path.mkdir(exist_ok=True)
        
        self.log_path = log_path.joinpath(log_name)
    
        for path in [path for path in self.log_path.parents][::-1] : 
            path.mkdir(exist_ok=True)
            
        self.check_exists()
        

    def __call__(self, log, end = '\n') : 
        now = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        print(f'[{now}] {log}', end = end)
        with open(self.log_path, 'a') as f :
            f.write(f'[{now}] {log}{end}')

    def check_exists(self) : 
        try :  
            if self.log_path.exists() : 
                shutil.rmtree(self.log_path)
        except : 
                os.remove(self.log_path)    