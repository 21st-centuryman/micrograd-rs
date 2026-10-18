use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Ident, LitInt, Token,
    ext::IdentExt,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

const LOSS_SLACK: usize = 32;

struct MlpInput {
    crate_path: Ident,
    sizes: Vec<usize>,
}

impl Parse for MlpInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let crate_path = input.call(Ident::parse_any)?;
        input.parse::<Token![;]>()?;
        let lits = Punctuated::<LitInt, Token![,]>::parse_terminated(input)?;
        let sizes = lits.iter().map(|l| l.base10_parse()).collect::<syn::Result<Vec<usize>>>()?;
        if sizes.len() < 2 {
            return Err(input.error("At least 2 sizes required (input and output)"));
        }
        Ok(MlpInput { crate_path, sizes })
    }
}

#[proc_macro]
pub fn generate_mlp(input: TokenStream) -> TokenStream {
    let MlpInput { crate_path, sizes } = parse_macro_input!(input as MlpInput);

    let num_layers = sizes.len() - 1;
    let layer_names: Vec<Ident> = (1..=num_layers).map(|i| format_ident!("l{}", i)).collect();
    let act_params: Vec<Ident> = (1..=num_layers).map(|i| format_ident!("act{}", i)).collect();
    let inn = &sizes[..num_layers];
    let out = &sizes[1..];

    let layer_fields = layer_names.iter().zip(inn.iter().zip(out)).map(|(li, (inn, out))| {
        quote! { #li: #crate_path::nn::Layer<'t, #inn, #out> }
    });

    let layer_inits = layer_names.iter().zip(&act_params).map(|(li, act)| {
        quote! { #li: #crate_path::nn::Layer::new(tape, #act) }
    });

    let forward_expr = layer_names.iter().fold(quote! { x }, |acc, layer| {
        quote! { self.#layer.forward(& #acc) }
    });

    let params_expr = {
        let first_layer = &layer_names[0];
        layer_names[1..].iter().fold(quote! { self.#first_layer.parameters() }, |acc, layer| {
            quote! { #acc.chain(self.#layer.parameters()) }
        })
    };

    // Tape size from layer sizes
    let params: usize = inn.iter().zip(out).map(|(p, n)| n * (p + 1)).sum();
    let forward: usize = sizes[0] + inn.iter().zip(out).map(|(p, n)| n * 2 * p + n).sum::<usize>();
    let tape_len = params + (forward + LOSS_SLACK).max(2 * params + LOSS_SLACK);

    let first_size = sizes[0];
    let last_size = sizes[num_layers];

    let first_layer = &layer_names[0];
    let rest_layers = &layer_names[1..];

    quote! {
        pub const TAPE_LEN: usize = #tape_len;
        pub type MLPTape = #crate_path::engine::Tape<[::std::cell::Cell<#crate_path::engine::ValueData>; TAPE_LEN]>;

        pub struct MLP<'t> {
            #( #layer_fields ),*
        }

        impl<'t> MLP<'t> {
            pub fn new(tape: &'t #crate_path::engine::Tape, #( #act_params: #crate_path::engine::Activations ),* ) -> Self {
                Self { #( #layer_inits ),* }
            }

            pub fn forward(&self, x: &[#crate_path::engine::Value<'t>; #first_size]) -> [#crate_path::engine::Value<'t>; #last_size] {
                #forward_expr
            }

            pub fn parameters(&self) -> impl Iterator<Item = &#crate_path::engine::Value<'t>> {
                #params_expr
            }
        }

        // Generated Debug impl
        impl std::fmt::Debug for MLP<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "MLP of [")?;
                write!(f, "{:?}", self.#first_layer)?;
                #( write!(f, ", {:?}", self.#rest_layers)?; )*
                write!(f, "]")
            }
        }
    }
    .into()
}
