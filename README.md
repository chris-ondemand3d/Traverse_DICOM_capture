1. Bone opacity 중간 정도 값을 생성
2. Medical Data에 soft tissue가 삼하게 안맞는 부분(다른 parameter)
3. Connected 분류
    # Maxilla -> Most largest connected components
    # Mandible -> at most 2 components
    # Spine 혹은 위에서 봤을때, 영역설정(가능하면 manual labeling, L. Condyle, R. Condyle 도 manual labeling 

4. png, txt파일읽어서 photo 확인하는 python code 
5. Relures에 scandirectory.loaddicom function수정
    # multiframe관련
    # sort series based on position, orientation 
    # make nifti file 부분
    # Update tableview in QML
    


