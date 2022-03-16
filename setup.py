from setuptools import setup, find_packages

setup(
    name='check_your_heuristic',
    version='0.0.2',
    license='MIT',
    description='Small library which aim is to check your dataset for being solved by simple heuristics. Multilingual!',
    author='Tatiana Iazykova, Olga Bystrova, Denis Kapelyushnik',
    author_email='tania_yazykova@bk.ru',
    keywords=['heuristics', 'rule-based', 'language models', 'natural language understanding', 'nlp'],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'run-base-case = check_your_heuristic.main:base_case',
            'run-multirc-case = check_your_heuristic.main:multirc_case',
            'run-record-case = check_your_heuristic.main:record_case',
            'run-wordincontext-case = check_your_heuristic.main:wordincontext_case',
        ],
    },
    install_requires=[
          'numpy>=1.19.0',
          'pandas>=1.1.5',
          'scikit-learn>=0.24.2',
          'pymorphy2>=0.9.1',
          'pyyaml',
          'nptyping>=1.4.4',
          'parameterized>=0.8.1',
          'openpyxl>=3.0.9',
          'matplotlib>=3.4.3',
          'seaborn>=0.11.0'
      ],
    classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    ],
    )
