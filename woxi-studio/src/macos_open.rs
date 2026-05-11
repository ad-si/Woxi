#![cfg(target_os = "macos")]

//! macOS file-open Apple Event handler.
//!
//! Files opened via Finder's "Open With…" arrive as the `kAEOpenDocuments`
//! Apple Event. AppKit's default handler for it calls `application:openURLs:`
//! on the `NSApplicationDelegate`. iced 0.14 and winit 0.30 don't implement
//! that method on the delegate they install, so AppKit shows "<App> cannot
//! open files in the … format" and the app never sees the request.
//!
//! Workaround: observe `NSApplicationWillFinishLaunchingNotification` —
//! the first notification after iced has installed its delegate and which
//! fires before any queued Apple Events are dispatched. In the observer,
//! dynamically add `application:openURLs:` to the delegate's class with
//! `class_addMethod`. AppKit's default `kAEOpenDocuments` dispatcher then
//! routes through us. The extracted paths land in a global queue that
//! iced's `Subscription` polls.

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, Sel};
use objc2::{AllocAnyThread, class, define_class, msg_send, sel};
use objc2_foundation::{NSObject, NSString};

static PENDING: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

pub fn take_pending() -> Vec<PathBuf> {
  std::mem::take(&mut *PENDING.lock().unwrap())
}

extern "C" fn application_open_urls(
  _self: *mut AnyObject,
  _cmd: Sel,
  _app: *mut AnyObject,
  urls: *mut AnyObject,
) {
  let mut paths = Vec::new();
  unsafe {
    if !urls.is_null() {
      let count: usize = msg_send![urls, count];
      for i in 0..count {
        let url: *mut AnyObject = msg_send![urls, objectAtIndex: i];
        if url.is_null() {
          continue;
        }
        let ns_path: *mut NSString = msg_send![url, path];
        if ns_path.is_null() {
          continue;
        }
        paths.push(PathBuf::from((*ns_path).to_string()));
      }
    }
  }
  if !paths.is_empty() {
    PENDING.lock().unwrap().extend(paths);
  }
}

unsafe extern "C" {
  fn class_addMethod(
    cls: *mut AnyClass,
    name: Sel,
    imp: extern "C" fn(),
    types: *const u8,
  ) -> bool;
}

define_class!(
  #[unsafe(super = NSObject)]
  struct WoxiOpenObserver;

  impl WoxiOpenObserver {
    #[unsafe(method(patchDelegate:))]
    fn patch_delegate(&self, _notification: &AnyObject) {
      unsafe {
        let app: *mut AnyObject =
          msg_send![class!(NSApplication), sharedApplication];
        let delegate: *mut AnyObject = msg_send![app, delegate];
        if delegate.is_null() {
          return;
        }
        let delegate_class: *mut AnyClass = msg_send![delegate, class];
        // Encoding for `-(void)application:(id)app openURLs:(id)urls`.
        let types = b"v@:@@\0".as_ptr();
        let imp: extern "C" fn() =
          std::mem::transmute(application_open_urls as *const c_void);
        let _added = class_addMethod(
          delegate_class,
          sel!(application:openURLs:),
          imp,
          types,
        );
      }
    }
  }
);

impl WoxiOpenObserver {
  fn new() -> Retained<Self> {
    unsafe { msg_send![Self::alloc(), init] }
  }
}

/// Register the observer. Call once, before `iced::application(...).run()`.
pub fn register() {
  unsafe {
    let observer = WoxiOpenObserver::new();
    // Leak: NSNotificationCenter holds a weak reference; the observer must
    // outlive the process.
    let observer_ptr: *mut WoxiOpenObserver = Retained::into_raw(observer);

    let center: *mut AnyObject =
      msg_send![class!(NSNotificationCenter), defaultCenter];
    let name =
      NSString::from_str("NSApplicationWillFinishLaunchingNotification");
    let _: () = msg_send![
      center,
      addObserver: &*observer_ptr,
      selector: sel!(patchDelegate:),
      name: &*name,
      object: std::ptr::null::<AnyObject>(),
    ];
  }
}
