from distutils.core import setup

setup(
    name='oarl',
    version='2.0.0',
    author='Simon Haxby',
    author_email='simon.haxby@gmail.com',
    url='https://github.com/avicennax/open-ai-fun',
    packages=['oarl'],
    license='MIT',
    description="An RL odyssey with Open-AI's gym framework and other friendos.",
    long_description=open('README.md').read(),
    install_requires=open('requirements.txt').read().split('\n')[:-1]
)
