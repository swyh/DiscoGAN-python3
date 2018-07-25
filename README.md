# DiscoGAN_python3

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
- 기존 DiscoGAN github 참조

### Training
- [기존 코드](https://github.com/SKTBrain/DiscoGAN)를 다운로드한 뒤, python3로 작성된 image_translation.py, dataset.py를 덮어쓰기
- 실행 : task_name에서 작은 따음표를 제외한 뒤 입력
    
      $ python ./discogan/image_translation.py --task_name=edges2handbags
    

### Test (개발 중)
- 저장된 model을 불러와 datasetA => datasetB 생성

      $ python ./discogan/load.py
    
## Add it in the future
- traing checkpoint model 이어서 시키기
- image size에 따라 layer 재구성
- python3 code로 정리
