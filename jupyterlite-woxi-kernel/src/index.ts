import {
  JupyterLiteServer,
  JupyterLiteServerPlugin,
} from '@jupyterlite/server';

import { IKernel, IKernelSpecs } from '@jupyterlite/kernel';

import { WoxiKernel } from './kernel';

const server_kernel: JupyterLiteServerPlugin<void> = {
  id: '@woxi/jupyterlite-woxi-kernel:kernel',
  autoStart: true,
  requires: [IKernelSpecs],
  activate: (app: JupyterLiteServer, kernelspecs: IKernelSpecs) => {
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

const plugins: JupyterLiteServerPlugin<any>[] = [server_kernel];
export default plugins;
