# DiscoGAN_python3
python2로 작성된 code를 python3로 변환, test 기능 추가<br>

Official paper : [https://arxiv.org/pdf/1703.05192.pdf](https://arxiv.org/pdf/1703.05192.pdf) <br>
Official implement : [https://github.com/SKTBrain/DiscoGAN](https://github.com/SKTBrain/DiscoGAN)

## Prerequisites
- Python 3.6
- PyTorch
- Numpy/Scipy/Pandas
- Progressbar
- OpenCV

## Execution

### Dataset Download
기존 DiscoGAN github 참조

### Training
[기존 코드](https://github.com/SKTBrain/DiscoGAN)를 다운로드한 뒤, discogan 폴더 안에 python3로 작성된 위 code로 덮어쓰기<br>

실행 : task_name에서 작은 따음표를 제외한 뒤 입력
    
    $ python ./discogan/image_translation.py --task_name=edges2handbags
    

### Test
저장된 model을 불러와 반대 dataset 생성

    $ python ./discogan/load.py --epoch=-3.0 --type=A
    
이 외에도 result_path, input_path, model_path 설정 가능
