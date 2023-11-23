from setuptools import setup,find_packages

setup(
    
    name = 'BlueSimulator', 
    
    
    version = '0.1.0',
    
    
    description = 'An MAB python package for computing environment for Next-Best-Actions and many other text features.',
    
        
    
    py_modules = ["NeuralUCB","NeuralThompsonSampling"],
    packages = find_packages(),
    
    
    author = 'Yash Kumar Singh Jha',
    author_email = 'ae19b016@smail.iitm.ac.in',
    
    
    long_description = open('README.md').read() + '\n\n' + open('CHANGELOG.md').read(),
    long_description_content_type = "text/markdown",
    
    
    url='https://github.com/YashIITM/Blue-Simulator',
    
    
    include_package_data=True,
    
    
    keywords = ['Multi Armed Bandits', 'Data Science', 'Bandits', 'LinUCB' , 'LinUCBVI', 'NeuralUCB'],
    
)
