import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
} from '@jupyterlab/application';

import { IKernel, IKernelSpecs } from '@jupyterlite/services';

import { WoxiKernel } from './kernel';

const kernel: JupyterFrontEndPlugin<void> = {
  id: '@woxi/jupyterlite-woxi-kernel:kernel',
  description: 'A plugin providing the Woxi (Wolfram Language) kernel.',
  autoStart: true,
  requires: [IKernelSpecs],
  activate: (app: JupyterFrontEnd, kernelspecs: IKernelSpecs) => {
    kernelspecs.register({
      spec: {
        name: 'woxi',
        display_name: 'Woxi (Wolfram Language)',
        language: 'wolfram',
        argv: [],
        resources: {
          'logo-32x32': '',
          'logo-64x64': '',
        },
      },
      create: async (options: IKernel.IOptions): Promise<IKernel> => {
        return new WoxiKernel(options);
      },
    });
  },
};

const plugins: JupyterFrontEndPlugin<any>[] = [kernel];
export default plugins;
