from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    '''
    This function will output the required lib list
    '''
    requirements=[]
    with open(file_path) as obj_file:
        requirements = obj_file.readlines()
        requirements = [req.replace("\n"," ") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(  
    name='MLProject_2',
    version= '0.0.1',
    author= 'Jernes',
    author_email='georgenjernes@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)